import jax
from jax import vmap,jit,pmap
import jax.numpy as jnp
from .time_step import time_step_dict
from ..solver_1D import flux as flux_1D
from ..solver_1D import aux_func as aux_func_1D
from ..solver_2D import flux #as flux
from ..solver_2D import aux_func #as aux_func
from ..model import thermo_model,reaction_model,transport_model
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from ..parallel.grid_partion import split_and_distribute_grid
from functools import partial
from tqdm import tqdm
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

from threading import Thread
from queue import Queue
import queue
from typing import Dict, Any, List, Optional
from typing import Union, Iterable
import re

    
def set_rhs(dim,reaction_config,source_config=None,is_parallel=False,is_amr=False):
    if dim == '1D':
        if is_parallel:
            boundary_conditions = parallel_boundary.boundary_conditions_1D
        else:
            boundary_conditions = boundary.boundary_conditions_1D
        def flux_func(U, aux, dx, dy, dt, theta):
            U_with_ghost,aux_with_ghost = boundary_conditions(U,aux,theta)
            rhs = dt*(flux_1D.total_flux(U_with_ghost,aux_with_ghost,dx))
            return rhs
        update_func = aux_func_1D.update_aux
         
    if dim == '2D':
        if is_parallel:
            boundary_conditions = parallel_boundary.boundary_conditions_2D
        else:
            boundary_conditions = boundary.boundary_conditions_2D
        if is_amr:
            @partial(vmap,in_axes=(0,0,None,None,None,None))
            def flux_func(U, aux, dx, dy, dt, theta):
                physical_rhs = dt*(flux.total_flux(U,aux,dx,dy))
                return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))
            update_func = vmap(aux_func.update_aux,in_axes=(0,0))
        else:
            def flux_func(U, aux, dx, dy, dt, theta):
                U_with_ghost,aux_with_ghost = boundary_conditions(U,aux,theta)
                rhs = dt*(flux.total_flux(U_with_ghost,aux_with_ghost,dx,dy))
                return rhs
            update_func = aux_func.update_aux
            
    if reaction_config['is_detailed_chemistry']:
        if source_config is not None:
            source_func  = source_config['self_defined_source_terms']
        else:
            source_func = None
    else:
        if source_config is not None:
            user_source_func  = source_config['self_defined_source_terms']
            def source_func(U, aux, dt, theta):
                return user_source_func(U,aux,theta)*dt + reaction_model.reaction_source_terms(U,aux,dt,theta)
        else:
            if ('self_defined_reaction_source_terms' not in reaction_config) or (reaction_config['self_defined_reaction_source_terms'] is None):
                source_func = None
            else:
                source_func = reaction_model.reaction_source_terms

    if is_amr and (source_func is not None):
        temp_source_func = source_func
        @partial(vmap,in_axes=(0,0,None,None))
        def source_func(U, aux, dt, theta):
            return temp_source_func(U[:,3:-3,3:-3],aux[:,3:-3,3:-3],dt,theta)
    return flux_func, update_func, source_func

def set_advance_func(dim,flux_config,reaction_config,time_control,is_amr,flux_func,update_func,source_func):
    is_detailed_chemistry = reaction_config['is_detailed_chemistry']
    solver_type = flux_config['solver_type']
    time_scheme = time_control['temporal_evolution_scheme'] + (is_amr)*('_amr')

    if source_func is None:
        if is_amr:
            def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, flux_func, update_func)
        else:
            def advance_flux(U,aux,dx,dy,dt,theta=None):
                return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
    else:
        if solver_type == 'flux_splitting':
            def rhs_func(U,aux,dx,dy,dt,theta=None):
                return flux_func(U,aux,dx,dy,dt,theta) + source_func(U,aux,dt,theta)
            if is_amr:
                def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                    return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, rhs_func, update_func)
            else:
                def advance_flux(U,aux,dx,dy,dt,theta=None):
                    return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,rhs_func,update_func)
        if solver_type == 'godunov':
            def wrapped_source_func(U,aux,dx,dy,dt,theta=None):
                return source_func(U,aux,dt,theta)
            if is_amr:
                def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                    field_adv = time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, flux_func, update_func)
                    return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, field_adv, ref_blk_info, theta, wrapped_source_func, update_func)
            else:
                def advance_flux(U,aux,dx,dy,dt,theta=None):
                    U_adv,aux_adv = time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
                    return time_step_dict[time_scheme](U_adv,aux_adv,dx,dy,dt,theta,wrapped_source_func,update_func)

    if is_detailed_chemistry:
        if is_amr:
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                blk_data = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None)
                U, aux = blk_data[:,0:-2],blk_data[:,-2:]
                dU = vmap(reaction_model.reaction_source_terms,in_axes=(0,0,None,None))(U,aux,dt,theta)
                U = U + dU
                aux = update_func(U, aux)
                return jnp.concatenate([U,aux],axis=1)
        else:
            def advance_one_step(U,aux,dx,dy,dt,theta=None):
                U, aux = advance_flux(U,aux,dx,dy,dt,theta)
                dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
                U = U + dU
                aux = update_func(U, aux)
                return U, aux
    else:
        advance_one_step = advance_flux
    return advance_one_step

class H5Saver:
    def __init__(self, filepath: str | Path, data_dim: str):
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "a")
        self.io_queue = Queue(maxsize=10)  # 限制队列大小，避免内存溢出
        self.io_thread = Thread(target=self._io_worker, daemon=True)
        self.io_thread.start()  # 启动后台I/O线程

        # 初始化get_prim（保持不变）
        if data_dim == '1D':
            def get_prim(U, aux):
                aux = aux_func_1D.update_aux(U, aux)
                rho, u, Y, p, a = aux_func_1D.U_to_prim(U, aux)
                q = {'density': rho[0], 'x-velocity': u[0], 'pressure': p[0], 'tempreature': aux[1], 'species': Y}
                return q
            self.get_prim = get_prim
        elif data_dim == '2D':
            def get_prim(U, aux):
                aux = aux_func.update_aux(U, aux)
                rho, u, v, Y, p, a = aux_func.U_to_prim(U, aux)
                q = {'density': rho[0], 'x-velocity': u[0], 'y-velocity': v[0], 'pressure': p[0], 'tempreature': aux[1], 'species': Y}
                return q
            self.get_prim = get_prim

    def _io_worker(self):
        """后台I/O工作线程，处理队列中的写入任务"""
        while True:
            try:
                task = self.io_queue.get(timeout=1)  # 超时退出机制
                if task is None:  # 终止信号
                    break
                func, args, kwargs = task
                func(*args, **kwargs)  # 执行实际写入操作
                self.io_queue.task_done()
            except queue.Empty:
                continue

    def _save_to_disk(self, key: str, t: float, step: int, arrays: Dict[str, np.ndarray]):
        """实际执行写入磁盘的操作（在后台线程中运行）"""
        with h5py.File(self.filepath, "a") as f:  # 每次写入重新打开文件，避免线程安全问题
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.attrs["time"] = t
            grp.attrs["step"] = step
            for name, arr in arrays.items():
                grp.create_dataset(name, data=arr, compression="gzip")
            f.flush()

    def save(self, save_step: int, t: float, step: int,** arrays: jnp.ndarray):
        """优化后的单设备保存：异步传输+后台I/O"""
        key = f"save_step{save_step}"
        
        # 修正：使用 jax.device_get 而非直接 device_get
        arrays_cpu_future = {name: jax.device_get(arr) for name, arr in arrays.items()}
        
        # 将写入任务放入队列，由后台线程执行
        self.io_queue.put((self._save_to_disk, (key, t, step, arrays_cpu_future), {}))

    def _parallel_save_to_disk(self, base_key: str, t: float, step: int, sharded_arrays: List[Dict[str, np.ndarray]]):
        """并行模式下的实际写入操作（后台线程执行）"""
        with h5py.File(self.filepath, "a") as f:
            if base_key in f:
                del f[base_key]
            # 遍历每个设备的分片数据
            for device_id, arrays in enumerate(sharded_arrays):
                device_key = f"{base_key}_device{device_id}"
                grp = f.create_group(device_key)
                grp.attrs["time"] = t
                grp.attrs["step"] = step
                grp.attrs["device_id"] = device_id
                for name, arr in arrays.items():
                    grp.create_dataset(name, data=arr, compression="gzip")
            f.flush()

    def parallel_save(self, save_step: int, t: float, step: int, **arrays: jnp.ndarray):
        """优化后的多设备保存：直接按设备索引分片+异步传输+后台I/O"""
        base_key = f"save_step{save_step}"
        num_devices = jax.device_count()  # 直接获取设备数量
        
        # 修正：使用 jax.device_get 而非直接 device_get
        sharded_arrays = []
        for i in range(num_devices):
            device_arrays = {name: jax.device_get(arr[i]) for name, arr in arrays.items()}
            sharded_arrays.append(device_arrays)
        
        # 后台线程执行写入
        self.io_queue.put((self._parallel_save_to_disk, (base_key, t, step, sharded_arrays), {}))


    def list_snapshots(self):
        "返回所有存储的时间步名称"
        return list(self.file.keys())

    def collect(self, concat_axis: int = 1):
        """
        在原 HDF5 文件里，把每个时间步的所有设备 group concat 后
        写入新的 group，命名为 save_step{step}。
        """
        with h5py.File(self.filepath, 'a') as f:
            # step -> list of group names
            step_groups_map = defaultdict(list)

            # 遍历所有 group
            for gname in f.keys():
                if gname.startswith("save_step") and "_device" in gname:
                    step_str = gname.split("_device")[0]  # e.g., "save_step1"
                    step_groups_map[step_str].append(gname)

            # 对每个 step 按设备号排序
            for step, groups in step_groups_map.items():
                groups.sort(key=lambda x: int(x.split("_device")[-1]))

            # 遍历每个 step，将设备数据 concat 后写入新 group
            for step, groups in step_groups_map.items():
                collected_step = {}
                attrs = {}

                # 收集每个 key 的数组，并保留 attrs
                for gname in groups:
                    grp = f[gname]
                    # 记录 attrs（以最后一个设备为准）
                    attrs = dict(grp.attrs)
                    for key in grp.keys():
                        arr = np.array(grp[key])
                        if key not in collected_step:
                            collected_step[key] = [arr]
                        else:
                            collected_step[key].append(arr)

                # concat 每个 key
                for key in collected_step:
                    collected_step[key] = np.concatenate(collected_step[key], axis=concat_axis)

                # 新 group 名称
                new_group_name = step
                if new_group_name in f:
                    del f[new_group_name]  # 避免冲突
                new_grp = f.create_group(new_group_name)

                # 写 attrs
                for k, v in attrs.items():
                    new_grp.attrs[k] = v

                # 写数据
                for key, arr in collected_step.items():
                    new_grp.create_dataset(key, data=arr, compression="gzip")

                for gname in groups:
                    if gname in f:  # 双重检查，防止组已被意外删除
                       del f[gname]
            f.flush()

    def collect_sub(self, step: Union[int, Iterable[int]], concat_axis: int = 1):
        """
        在原 HDF5 文件里，将指定 step（单个或多个）的所有设备 group concat 后
        写入新的 group，命名为 save_step{step}，并删除原设备分组。
    
        参数:
            step: 要处理的时间步（单个整数或整数数组/列表，如 300000 或 [300000, 600000]）
            concat_axis: 拼接的轴，默认值为 1
        """
        # 处理单个整数（包括 Python 原生 int、NumPy/JAX 整数）
        if isinstance(step, (int, np.integer)):
            steps = [int(step)]
        else:
            # 关键修改：将可迭代对象（JAX/NumPy 数组）转换为 Python 列表
            try:
                # 先转换为 NumPy 数组，再转为列表（兼容 JAX 数组）
                step_list = np.asarray(step).tolist()
            except:
                step_list = list(step)  # 常规可迭代对象直接转列表
        
            # 过滤并转换为整数
            steps = []
            for s in step_list:
                try:
                    # 尝试将元素转换为整数（处理 float 型整数如 400.0）
                    int_s = int(s)
                    if int_s == s:  # 确保转换后值不变（排除非整数如 400.5）
                        steps.append(int_s)
                except (ValueError, TypeError):
                    continue  # 跳过无法转换为整数的元素
        
            if not steps:
                raise ValueError("step 必须包含至少一个整数")
        #print("最终处理的 steps 列表：", steps)

        with h5py.File(self.filepath, 'a') as f:
            # 遍历每个需要处理的 step
            for s in steps:
                target_step_prefix = f"save_step{s}"
                target_step_str = target_step_prefix

                # 收集当前 step 对应的所有设备分组
                pattern = re.compile(rf"^{target_step_prefix}_device\d+$")
                device_groups = [
                    gname for gname in f.keys()
                    if pattern.match(gname)  # 只匹配完全符合格式的分组
                   ]

                # 检查是否存在设备分组
                if not device_groups:
                    print(f"警告：未找到 step {s} 对应的设备分组，已跳过")
                    continue  # 跳过不存在的 step

                # 按设备号排序
                device_groups.sort(key=lambda x: int(x.split("_device")[-1]))

                # 收集数据和属性
                collected_step = defaultdict(list)
                attrs = {}

                for gname in device_groups:
                    grp = f[gname]
                    attrs = dict(grp.attrs)  # 保留最后一个设备的属性
                    for key in grp.keys():
                        arr = np.array(grp[key])
                        collected_step[key].append(arr)

                # 拼接数据
                for key in collected_step:
                    collected_step[key] = np.concatenate(collected_step[key], axis=concat_axis)

                # 创建新组（删除已有组避免冲突）
                if target_step_str in f:
                    del f[target_step_str]
                new_grp = f.create_group(target_step_str)

                # 写入属性和数据
                for k, v in attrs.items():
                    new_grp.attrs[k] = v
                for key, arr in collected_step.items():
                    new_grp.create_dataset(key, data=arr, compression="gzip")

                # 删除原设备分组
                for gname in device_groups:
                    if gname in f:
                        del f[gname]

            f.flush()

    def load(self, key: str):
        "加载指定时间步的数据，返回 dict"
        if not self.file or not self.file.id.valid:
            self.file = h5py.File(self.filepath, 'r')
        grp = self.file[key]
        out = {}
        for name in grp.keys():
            out[name] = jnp.array(grp[name][:])
        U_init = jnp.array(out['u'])
        gamma_init = jnp.full_like(U_init[0:1],1.4)
        T_init = jnp.full_like(gamma_init,500.0)
        aux_init = jnp.concatenate([gamma_init,T_init],axis=0)
        prim = self.get_prim(U_init,aux_init)
        meta = dict(grp.attrs)
        time = meta['time']
        self.file.close()
        return time, prim

    def load_all(self):
        # 如果文件已经关闭就重新以只读方式打开
        if not self.file or not self.file.id.valid:
            self.file = h5py.File(self.filepath, 'r')
        def extract_step(s):
            # 去掉非数字字符，只保留数字
            digits = ''.join(ch for ch in s if ch.isdigit())
            return int(digits)

        steps = sorted(self.file.keys(), key=extract_step)
        # 按 step 排序
        #steps = sorted(self.file.keys(), key=lambda x: int(x))
    
        # 先读取第一个 step，确定有哪些变量
        first_group = self.file[steps[0]]
        var_names = list(first_group.keys())
    
        # 为每个变量收集所有 step 的数据
        data = {name: [] for name in var_names}
        meta_names = list(dict(first_group.attrs).keys())
        meta = {name: [] for name in meta_names}
        for step in steps:
            grp = self.file[step]
            mt = dict(grp.attrs)
            for name in var_names:
                data[name].append(np.array(grp[name]))
            for name in meta_names:
                meta[name].append(np.array(mt[name])) 
    
        # 在第0维堆叠
        for name in var_names:
            data[name] = np.stack(data[name], axis=0)
        for name in meta_names:
            meta[name] = np.stack(meta[name],axis=0)

        U_init = jnp.array(data['u'])
        gamma_init = jnp.full_like(U_init[:,0:1],1.4)
        T_init = jnp.full_like(gamma_init,500.0)
        aux_init = jnp.concatenate([gamma_init,T_init],axis=1)
        prim = vmap(self.get_prim,in_axes=(0,0))(U_init,aux_init)
        time = meta['time']
        self.file.close()
        return time,prim

    def close(self):
        self.file.close()



class H5Saver_old:
    def __init__(self, filepath: str | Path, data_dim: str):
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "a")
        
        if data_dim == '1D':
            def get_prim(U,aux):
                aux = aux_func_1D.update_aux(U,aux)
                rho,u,Y,p,a = aux_func_1D.U_to_prim(U,aux)
                q = {'density':rho[0],
                     'x-velocity':u[0],
                     'pressure':p[0],
                     'tempreature':aux[1],
                     'species':Y}
                return q
            self.get_prim = get_prim
        if data_dim == '2D':
            def get_prim(U,aux):
                aux = aux_func.update_aux(U,aux)
                rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
                q = {'density':rho[0],
                     'x-velocity':u[0],
                     'y-velocity':v[0],
                     'pressure':p[0],
                     'tempreature':aux[1],
                     'species':Y}
                return q
            self.get_prim = get_prim

    def save(self, save_step: int, t: float, step: int, **arrays: jnp.ndarray):
        """
        保存一个时间步的数据
        参数:
            t: 当前时间
            step: 当前步数
            arrays: 需要保存的 JAX arrays，形式为 name=array
        """
        key = f"save_step{save_step}"
        # 避免 name already exists
        if key in self.file:
            del self.file[key]
        grp = self.file.create_group(key)
        #grp = self.file.create_group(f"save_step{save_step}")
        grp.attrs["time"] = t
        grp.attrs["step"] = step
        for name, arr in arrays.items():
            arr_cpu = np.array(arr.block_until_ready())  # 拉回CPU
            grp.create_dataset(name, data=arr_cpu, compression="gzip")
        self.file.flush()  # 立即写盘，防止崩溃丢数据

    def parallel_save(self, save_step: int, t: float, step: int, **arrays: jnp.ndarray):
        key = f"save_step{save_step}"
        if key in self.file:
            del self.file[key]

        @partial(pmap,axis_name='x',in_axes=(None,0))
        def save_fcn(key, arr):
            device_id = jax.lax.axis_index('x')
            key = key + f"_device{device_id}"
            grp = self.file.create_group(key)
            #grp = self.file.create_group(f"save_step{save_step}")
            grp.attrs["time"] = t
            grp.attrs["step"] = step
            for name, arr in arrays.items():
                arr_cpu = np.array(arr.block_until_ready())  # 拉回CPU
                grp.create_dataset(name, data=arr_cpu, compression="gzip")
            self.file.flush()
        save_fcn(key,arrays)

    def list_snapshots(self):
        "返回所有存储的时间步名称"
        return list(self.file.keys())

    def collect(self, concat_axis: int = 1):
        """
        在原 HDF5 文件里，把每个时间步的所有设备 group concat 后
        写入新的 group，命名为 save_step{step}。
        """
        with h5py.File(self.filepath, 'a') as f:
            # step -> list of group names
            step_groups_map = defaultdict(list)

            # 遍历所有 group
            for gname in f.keys():
                if gname.startswith("save_step") and "_device" in gname:
                    step_str = gname.split("_device")[0]  # e.g., "save_step1"
                    step_groups_map[step_str].append(gname)

            # 对每个 step 按设备号排序
            for step, groups in step_groups_map.items():
                groups.sort(key=lambda x: int(x.split("_device")[-1]))

            # 遍历每个 step，将设备数据 concat 后写入新 group
            for step, groups in step_groups_map.items():
                collected_step = {}
                attrs = {}

                # 收集每个 key 的数组，并保留 attrs
                for gname in groups:
                    grp = f[gname]
                    # 记录 attrs（以最后一个设备为准）
                    attrs = dict(grp.attrs)
                    for key in grp.keys():
                        arr = np.array(grp[key])
                        if key not in collected_step:
                            collected_step[key] = [arr]
                        else:
                            collected_step[key].append(arr)

                # concat 每个 key
                for key in collected_step:
                    collected_step[key] = np.concatenate(collected_step[key], axis=concat_axis)

                # 新 group 名称
                new_group_name = step
                if new_group_name in f:
                    del f[new_group_name]  # 避免冲突
                new_grp = f.create_group(new_group_name)

                # 写 attrs
                for k, v in attrs.items():
                    new_grp.attrs[k] = v

                # 写数据
                for key, arr in collected_step.items():
                    new_grp.create_dataset(key, data=arr, compression="gzip")
            f.flush()

    def load(self, key: str):
        "加载指定时间步的数据，返回 dict"
        if not self.file or not self.file.id.valid:
            self.file = h5py.File(self.filepath, 'r')
        grp = self.file[key]
        out = {}
        for name in grp.keys():
            out[name] = jnp.array(grp[name][:])
        U_init = jnp.array(out['u'])
        gamma_init = jnp.full_like(U_init[0:1],1.4)
        T_init = jnp.full_like(gamma_init,500.0)
        aux_init = jnp.concatenate([gamma_init,T_init],axis=0)
        prim = self.get_prim(U_init,aux_init)
        meta = dict(grp.attrs)
        time = meta['time']
        self.file.close()
        return time, prim

    def load_all(self):
        # 如果文件已经关闭就重新以只读方式打开
        if not self.file or not self.file.id.valid:
            self.file = h5py.File(self.filepath, 'r')
        def extract_step(s):
            # 去掉非数字字符，只保留数字
            digits = ''.join(ch for ch in s if ch.isdigit())
            return int(digits)

        steps = sorted(self.file.keys(), key=extract_step)
        # 按 step 排序
        #steps = sorted(self.file.keys(), key=lambda x: int(x))
    
        # 先读取第一个 step，确定有哪些变量
        first_group = self.file[steps[0]]
        var_names = list(first_group.keys())
    
        # 为每个变量收集所有 step 的数据
        data = {name: [] for name in var_names}
        meta_names = list(dict(first_group.attrs).keys())
        meta = {name: [] for name in meta_names}
        for step in steps:
            grp = self.file[step]
            mt = dict(grp.attrs)
            for name in var_names:
                data[name].append(np.array(grp[name]))
            for name in meta_names:
                meta[name].append(np.array(mt[name])) 
    
        # 在第0维堆叠
        for name in var_names:
            data[name] = np.stack(data[name], axis=0)
        for name in meta_names:
            meta[name] = np.stack(meta[name],axis=0)

        U_init = jnp.array(data['u'])
        gamma_init = jnp.full_like(U_init[:,0:1],1.4)
        T_init = jnp.full_like(gamma_init,500.0)
        aux_init = jnp.concatenate([gamma_init,T_init],axis=1)
        prim = vmap(self.get_prim,in_axes=(0,0))(U_init,aux_init)
        time = meta['time']
        self.file.close()
        return time,prim

    def close(self):
        self.file.close()

class Simulator:
    def __init__(self,simulation_config):
        dim = simulation_config['dimension']
        grid_config = simulation_config['grid_config']
        thermo_config = simulation_config['thermo_config']
        reaction_config = simulation_config['reaction_config']
        if 'transport_config' in simulation_config:
            transport_config = simulation_config['transport_config']
        else:
            transport_config = None
        flux_config = simulation_config['flux_config']
        boundary_config = simulation_config['boundary_config']
        if 'source_config' in simulation_config:
            source_config = simulation_config['source_config']
        else:
            source_config = None
        if 'nondim_config' in simulation_config:
            nondim_config = simulation_config['nondim_config']
        else:
            nondim_config = None
        time_control = simulation_config['time_config']
        self.t_end = time_control['t_end']

        if 'solver_parameters' in simulation_config:
            theta = simulation_config['solver_parameters']
        else:
            theta = None
        self.theta = theta

        if 'computation_config' in simulation_config:
            computation_config = simulation_config['computation_config']
            if 'is_parallel' in computation_config:
                is_parallel = computation_config['is_parallel']
                num_devices = len(jax.devices())
                if is_parallel and (theta is not None):
                    assert 'PartitionDict' in computation_config, "A python dict specifying the partition axes of theta should be provided!"
                    PartitionDict = computation_config['PartitionDict']
                    #PartitionDict = {}
                    #for key, axis in raw_dict.items():
                        #n = len(jnp.shape(theta[key]))
                        #if n == 0:
                            #PartitionDict[key] = P()
                        #else:
                            #if axis == None:
                                #tup = tuple(None for i in range(n))
                            #else:
                                #tup = tuple('x' if i == axis else None for i in range(n))
                            #PartitionDict[key] = P(*tup)
                else:
                    PartitionDict = None#P()
            else:
                is_parallel = False

            if 'output_settings' in computation_config:
                output_settings = computation_config['output_settings']
                self.save_dt = output_settings['save_dt']
                self.results_path = output_settings['results_path']
            else:
                self.save_dt = self.t_end
                self.results_path = 'results.h5'
        else:
            is_parallel = False
            self.save_dt = self.t_end
            self.results_path = 'results.h5'
        self.is_parallel = is_parallel
        self.saver = H5Saver(self.results_path,dim)
            
        thermo_model.set_thermo(thermo_config,nondim_config,dim)
        reaction_model.set_reaction(reaction_config,nondim_config,dim)
        if dim == '1D':
            flux_1D.set_flux_solver(flux_config,transport_config,nondim_config)
        if dim == '2D':
            flux.set_flux_solver(flux_config,transport_config,nondim_config)
        boundary.set_boundary(boundary_config,dim)
        is_amr = False
        flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,is_parallel,is_amr)
        advance_func_body = set_advance_func(dim,flux_config,reaction_config,time_control,is_amr,flux_func,update_func,source_func)
                
        if 'dt' in time_control:
            dt = time_control['dt']
            if dim == '1D':
                dx = grid_config['dx']
                dy = None
            if dim == '2D':
                dx, dy = grid_config['dx'], grid_config['dy']
            def advance_func(U,aux,t,theta):
                U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
                t = t + dt
                return U,aux,t,dt

        if 'CFL' in time_control:
            CFL = time_control['CFL']
            if dim == '1D':
                dx = grid_config['dx']
                if is_parallel:
                    def CFL_1D(U,aux,dx,cfl=0.30):
                        _,u,_,_,a = aux_func_1D.U_to_prim(U,aux)
                        sx = jnp.abs(u) + a
                        return cfl*jax.lax.pmin(dx/sx,axis_name='x')
                else:
                    def CFL_1D(U,aux,dx,cfl=0.30):
                        _,u,_,_,a = aux_func_1D.U_to_prim(U,aux)
                        sx = jnp.abs(u) + a
                        return cfl*jnp.min(dx/sx)
                def advance_func(U,aux,t,theta):
                    dt = CFL_1D(U,aux,dx,CFL)
                    U, aux = advance_func_body(U,aux,dx,None,dt,theta)
                    return U, aux, t+dt, dt
            if dim == '2D':
                dx, dy = grid_config['dx'],grid_config['dy']
                if is_parallel:
                    def CFL_2D(U,aux,dx,dy,cfl=0.30):
                        _,u,v,_,_,a = aux_func.U_to_prim(U,aux)
                        sx = jnp.abs(u) + a
                        sy = jnp.abs(v) + a
                        dt = cfl*jax.lax.pmin(1/(sx/dx + sy/dy),axis_name='x')
                        return dt
                else:
                    def CFL_2D(U,aux,dx,dy,cfl=0.30):
                        _,u,v,_,_,a = aux_func.U_to_prim(U,aux)
                        sx = jnp.abs(u) + a
                        sy = jnp.abs(v) + a
                        dt = cfl*jnp.min(1/(sx/dx + sy/dy))
                        return dt
                def advance_func(U,aux,t,theta):
                    dt = CFL_2D(U,aux,dx,dy,CFL)
                    U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
                    return U, aux, t+dt, dt
        if is_parallel:
            #advance_func = jax.shard_map(advance_func,mesh=mesh,
                                         #in_specs=(P(None,'x',None),P(None,'x',None),P(),PartitionDict),
                                         #out_specs=(P(None,'x',None),P(None,'x',None),P(),P()),
                                         #check_vma=False)
            self.advance_func = pmap(advance_func,axis_name='x',in_axes=(0, 0, None, PartitionDict),out_axes=(0, 0, None, None))
        else:
            self.advance_func = jit(advance_func)

    def run(self,U_init,aux_init):
        advance_func = self.advance_func
        theta = self.theta
        U, aux = U_init,aux_init
        t = 0.0
        step = 0
        save_step = 1
        if self.is_parallel:
            U = split_and_distribute_grid(U)
            aux = split_and_distribute_grid(aux)
            self.saver.parallel_save(save_step=0, t=t, step=step, u=U)
        else:
            self.saver.save(save_step=0, t=t, step=step, u=U)
        save_dt = self.save_dt
        next_save_time = save_dt
        t_end = self.t_end
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        with tqdm(total=t_end, desc="Simulation", bar_format=bar_format) as pbar:
            while t < t_end:
                U, aux, t, dt = advance_func(U,aux,t,theta)
                step += 1

                if t >= next_save_time or t >= t_end:
                    if self.is_parallel:
                        self.saver.parallel_save(save_step, t, step, u=U)
                    else:
                        self.saver.save(save_step, t, step, u=U)
                    next_save_time += save_dt
                    save_step += 1

                if t > t_end:
                    pbar.n = pbar.total
                    pbar.refresh()
                else:
                    pbar.update(float(dt))
                    
        self.saver.close()
        return U, aux, t

def AMR_Simulator(simulation_config):
    dim = '2D'
    thermo_config = simulation_config['thermo_config']
    reaction_config = simulation_config['reaction_config']
    if 'transport_config' in simulation_config:
        transport_config = simulation_config['transport_config']
    else:
        transport_config = None
    flux_config = simulation_config['flux_config']
    boundary_config = simulation_config['boundary_config']
    if 'source_config' in simulation_config:
        source_config = simulation_config['source_config']
    else:
        source_config = None
    if 'nondim_config' in simulation_config:
        nondim_config = simulation_config['nondim_config']
    else:
        nondim_config = None
    time_control = simulation_config['time_config']
    if 'solver_parameters' in simulation_config:
            theta = simulation_config['solver_parameters']
    else:
        theta = None
    thermo_model.set_thermo(thermo_config,nondim_config,dim)
    reaction_model.set_reaction(reaction_config,nondim_config,dim)
    flux.set_flux_solver(flux_config,transport_config,nondim_config)
    boundary.set_boundary(boundary_config,dim)
    flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,False,True)
    advance_func_amr = set_advance_func(dim,flux_config,reaction_config,time_control,True,flux_func,update_func,source_func)
    flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,False,False)
    advance_func_body = set_advance_func(dim,flux_config,reaction_config,time_control,False,flux_func,update_func,source_func)
    def advance_func_base(blk_data,dx,dy,dt,theta=None):
        U, aux = blk_data[0,:-2],blk_data[0,-2:]
        U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
        blk_data = jnp.array([jnp.concatenate([U,aux],axis=0)])
        return blk_data
    return jit(advance_func_amr,static_argnames='level'),jit(advance_func_base)






































