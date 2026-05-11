"""
Patch for loading old Firedrake checkpoint files that fail after the
finat TensorProductElement family change (firedrake-fiat b0646c25).

Old checkpoints stored CG functions on extruded meshes with DG embedding
because TPE.family() was "TensorProductElement". New finat reports
"Lagrange" instead, so the checkpointing code thinks these are native
and hits an assertion when it finds DG-embedded data in the file.

This patch removes that assertion. The rest of the loading logic
(project from DG embed to target space) works fine without it.

Usage:
    import patch_checkpoint_load  # import before loading
    from firedrake import CheckpointFile

    with CheckpointFile("old_file.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        T = f.load_function(mesh, "Temperature")

See: https://github.com/firedrakeproject/firedrake/issues/4998
"""

import os
import firedrake.checkpointing as ckpt
from firedrake import Function
from firedrake.embedding import get_embedding_method_for_checkpointing
from pyop2 import op2


def _load_function_no_assert(self, mesh, name, idx=None):
    mesh = mesh.unique()
    tmesh = mesh.topology

    if name in self._get_mixed_function_name_mixed_function_space_name_map(mesh.name):
        V_name = self._get_mixed_function_name_mixed_function_space_name_map(mesh.name)[name]
        V = self._load_function_space(mesh, V_name)
        base_path = self._path_to_mixed_function(mesh.name, V_name, name)
        fsub_list = []
        for i, Vsub in enumerate(V):
            path = os.path.join(base_path, str(i))
            fsub_name = self.get_attr(path, ckpt.PREFIX + "_function")
            fsub = self.load_function(mesh, fsub_name, idx=idx)
            fsub_list.append(fsub)
        dat = op2.MixedDat(fsub.dat for fsub in fsub_list)
        return Function(V, val=dat, name=name)

    elif name in self._get_function_name_function_space_name_map(
        self._get_mesh_name_topology_name_map()[mesh.name], mesh.name
    ):
        tmesh_name = self._get_mesh_name_topology_name_map()[mesh.name]
        V_name = self._get_function_name_function_space_name_map(tmesh_name, mesh.name)[name]
        V = self._load_function_space(mesh, V_name)
        tV = V.topological
        path = self._path_to_function(tmesh_name, mesh.name, V_name, name)
        if ckpt.PREFIX_EMBEDDED in self.h5pyfile[path]:
            path = self._path_to_function_embedded(tmesh_name, mesh.name, V_name, name)
            _name = self.get_attr(path, ckpt.PREFIX_EMBEDDED + "_function")
            _f = self.load_function(mesh, _name, idx=idx)
            element = V.ufl_element()
            # assertion removed -- see module docstring
            method = get_embedding_method_for_checkpointing(element)
            f = Function(V, name=name)
            self._project_function_for_checkpointing(f, _f, method)
            return f
        else:
            tf_name = self.get_attr(path, ckpt.PREFIX + "_vec")
            tf = self._load_function_topology(tV.mesh(), tV.ufl_element(), tf_name, idx=idx)
            return Function(V, val=tf, name=name)
    else:
        raise RuntimeError(f"Function ({name}) not found in {self.filename}")


ckpt.CheckpointFile.load_function = _load_function_no_assert