# Example point clouds

There are two pairs of point clouds testing the robustness of the `ICP` algorithm.

## kg_pc8d
Scene with strong geometric features. From the `build` directory, execute:
```bash
./bin/icp_step_by_step
```

## kg_pc8d_wall
Scene with non-salient surface geometry. It highlights the benefit of utilizing the photometric information. To see what happens in the absence of color, change the `a` parameter in `src/step_by_step/ocl_icp_sbs.cpp` to a really small strictly positive number. From the `build` directory, execute:
```bash
./bin/icp_step_by_step kg_pc8d_wall
```
