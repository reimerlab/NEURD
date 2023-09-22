parameters = dict(
    spine_utils = dict(
        # --- shaft parameters ---
        soma_vertex_nullification = False,
        skeleton_endpoint_nullification = False,

        clusters_threshold = 6,
        smoothness_threshold = 0.08,
        shaft_mesh_volume_max = 0.4,
        shaft_close_hole_area_top_2_mean_max = 0.4,
        shaft_mesh_n_faces_min = 200,


        # spine filtering parameters
        spine_n_face_threshold_bare_min = 310,
        spine_sk_length_threshold_bare_min = 0.6,
        filter_by_volume_threshold_bare_min = 0.011,
        bbox_oriented_side_max_min_bare_min = 0.4,
        sdf_mean_min_bare_min = 0.1,
        spine_volume_to_spine_area_min_bare_min = 0.00002,

        # head neck segmentation
        head_ray_trace_min = 0.3,
        head_face_min = 400,
    )
)