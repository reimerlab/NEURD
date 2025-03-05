"""
Purpose
-------
Find all soma synapses (instance of finding all synapses on a submesh)

Pseudocode
----------
1) Download the full mesh id (decimated mesh)
2) Download the submesh face indices (soma face indices), soma_faces_idx
3) Synapse processing --
    a. Download all synapses for id
    b. Extract coordinates of all synapses
        call syn_coords
4) Build the KDTree for the mesh
    a. mesh_points = mesh.triangles_center (attribute of Trimesh object)
    b. build the KDTree
        KDTree(triangles_center)
5) query the KDTree with synapse coordinates
    distances,closest_node = kdtree_obj.query(syn_coords)
6) count the synapses (of each type) where closest node is on a soma indices
    np.where(closest_node in soma_faces_idx)

"""

from pykdtree.kdtree import KDTree

submesh_mesh_kdtree = KDTree(submesh_midpoints)
    #2) Query the fae midpoints of submesh against KDTree
distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)