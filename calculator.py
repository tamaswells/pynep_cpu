#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 14:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import contextlib
import os
import traceback
import shutil
import numpy as np
from structure import Structure
from nep_cpu import CpuNep

def clean_pycache(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if dir == "__pycache__":
                pycache_path = os.path.join(root, dir)
                shutil.rmtree(pycache_path)
                
class Nep3Calculator( ):

    def __init__(self, model_file="nep.txt"):
        super().__init__()
        if not isinstance(model_file, str):
            model_file = str(model_file )
        self.initialized = False

        if CpuNep is None:
            raise SystemError("Failed to import nep_cpu.\n")

        if os.path.exists(model_file):

            # model_file = model_file.encode("utf-8")

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    self.nep3 = CpuNep(model_file)
            self.element_list=self.nep3.get_element_list()
            self.type_dict = {e: i for i, e in enumerate(self.element_list)}

            self.initialized = True

    def compose_structures(self, structures:list[Structure]):
        group_size = []
        _types = []
        _boxs = []
        _positions = []
        if isinstance(structures, Structure):
            structures = [structures]
        for structure in structures:
            symbols = structure.elements
            _type = [self.type_dict[k] for k in symbols]
            _box = structure.cell.transpose(1, 0).reshape(-1).tolist()

            _position = structure.positions.transpose(1, 0).reshape(-1).tolist()
            _types.append(_type)
            _boxs.append(_box)
            _positions.append(_position)
            group_size.append(len(_type))
        return  _types, _boxs, _positions,group_size

    def calculate(self,structures:list[Structure]):
        if not self.initialized:
            return np.array([]),np.array([]),np.array([])

        _types, _boxs, _positions,group_size = self.compose_structures(structures)
        potentials, forces, virials = self.nep3.calculate(_types, _boxs, _positions)


        split_indices = np.cumsum(group_size)[:-1]
        #
        potentials=np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices)
        potentials_array = np.array(list(map(np.sum, split_potential_arrays)), dtype=np.float32)
        # print(potentials_array)

        # 处理每个force数组：reshape (3, -1) 和 transpose(1, 0)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]

        forces_array = np.vstack(reshaped_forces,dtype=np.float32)
        # print(forces_array)

        reshaped_virials = np.vstack([np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials],dtype=np.float32)

        virials_array = reshaped_virials[:,[0,4,8,1,5,6]]

        return potentials_array,forces_array,virials_array


    def get_descriptor(self,structure:Structure):
        if not self.initialized:
            return np.array([])
        symbols = structure.elements
        _type = [self.type_dict[k] for k in symbols]
        _box = structure.cell.transpose(1, 0).reshape(-1).tolist()

        _position = structure.positions.transpose(1, 0).reshape(-1).tolist()

        descriptor = self.nep3.get_descriptor(_type, _box, _position)

        descriptors_per_atom = np.array(descriptor,dtype=np.float32).reshape(-1, len(structure)).T

        return descriptors_per_atom

    def get_structures_descriptor(self,structures:list[Structure]):
        """
        返回的已经结构的描述符了 无需平均
        """
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)

        descriptor = self.nep3.get_structures_descriptor(_types, _boxs, _positions)

        return np.array(descriptor,dtype=np.float32)

    def get_structures_polarizability(self,structures:list[Structure]):
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)

        polarizability = self.nep3.get_structures_polarizability(_types, _boxs, _positions)

        return np.array(polarizability,dtype=np.float32)

    def get_structures_polarizability_atomic(self,structures:list[Structure]):
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)

        polarizability = self.nep3.get_structures_polarizability_atomic(_types, _boxs, _positions)

        return np.array(polarizability,dtype=np.float32)        

    def get_structures_dipole(self,structures:list[Structure]):
        if not self.initialized:
            return np.array([])
        _types, _boxs, _positions, group_size = self.compose_structures(structures)

        dipole = self.nep3.get_structures_dipole(_types, _boxs, _positions)

        return np.array(dipole,dtype=np.float32)

def run_nep3_calculator(nep_txt,structures,calculator_type,queue):
    try:
        nep3 = Nep3Calculator(nep_txt)
        if calculator_type == 'polarizability':
            result = nep3.get_structures_polarizability(structures)
        elif calculator_type == 'descriptor':
            result = nep3.get_structures_descriptor(structures)
        elif calculator_type == 'dipole':
            result = nep3.get_structures_dipole(structures)
        else:
            result = nep3.calculate(structures)
            
    except Exception as e:
        print(traceback.format_exc())
        result = np.array([])
    return result
    
def run_nep3_calculator_process(nep_txt,structures,calculator_type="calculate"):
    return run_nep3_calculator(nep_txt, structures,calculator_type)

if __name__ == '__main__':
    structures = Structure.read_multiple(r"train.xyz")
    nep = Nep3Calculator(r"nep.txt")
    print(nep.get_structures_polarizability_atomic(structures).mean(axis=0))
    print(nep.get_structures_polarizability_atomic(structures))

    current_dir = os.getcwd()
    clean_pycache(current_dir)
    