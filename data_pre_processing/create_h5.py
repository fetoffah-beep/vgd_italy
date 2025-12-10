# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:48:37 2025

@author: gmfet
"""

import xarray as xr
import h5py
import os
import numpy as np
data_path = r"C:\Users\gmfet\vgd_italy\data\static"
coord_names = {'y':
                   {'lat', 'latitude', 'y', 'northing', 'north'},
               'x':
                   {'lon', 'longitude', 'x', 'easting', 'east'}}

files = [file for file in os.listdir(data_path) if file.endswith('.nc')]

with h5py.File('../data/static_features.h5', 'a', libver="latest") as f:
    grp = f.require_group("static")

    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        ds = xr.open_dataset(
            file_path,
            engine="netcdf4",
            drop_variables=["ssm_noise", "spatial_ref", "band", "crs"],
            decode_cf=False,
            decode_times=False,
        )

        # Ensure CRS if rioxarray is available
        try:
            crs = ds.rio.crs
        except Exception:
            crs = None
        if not crs:
            try:
                ds = ds.rio.write_crs("EPSG:4326")
            except Exception:
                pass

        # find latitude/longitude coordinate names
        lat_name = next((c for c in ds.coords if c.lower() in coord_names["y"]), None)
        lon_name = next((c for c in ds.coords if c.lower() in coord_names["x"]), None)

        # chunk if possible
        chunks = {}
        if lat_name:
            chunks[lat_name] = 10
        if lon_name:
            chunks[lon_name] = 10
        if chunks:
            ds = ds.chunk(chunks)

        # create a subgroup per file (use filename without extension)
        file_group = grp.require_group(os.path.splitext(file_name)[0])

        # store each data variable as a dataset
        da = ds[var]
            # load variable values and ensure float32
            data = da.values.astype(np.float32)

            
            dset = file_group.create_dataset(
                var,
                data=data,
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )

            # copy attributes (ignore ones that fail)
            for k, v in da.attrs.items():
                try:
                    dset.attrs[k] = v
                except Exception:
                    pass

            if "long_name" not in dset.attrs:
                dset.attrs["long_name"] = da.attrs.get("long_name", var)

        ds.close()



# import xarray as xr
# import h5py
# import os

# # Path to where your predictor .nc files are stored
# data_dir = "/path/to/your/nc/files"
# output_file = "vgd_data.h5"

# # Create or overwrite the .h5 file
# with h5py.File(output_file, 'w') as h5f:

#     # Create groups
#     dynamic_group = h5f.create_group("dynamic")
#     static_group = h5f.create_group("static")

#     # Loop through files
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".nc"):
#             file_path = os.path.join(data_dir, file_name)

#             print(f"Processing {file_name} ...")

#             # Open with xarray lazily (doesn’t load data into memory yet)
#             ds = xr.open_dataset(file_path)

#             # Example: choose which group based on file name
#             if "precip" in file_name or "temp" in file_name or "tws" in file_name:
#                 group = dynamic_group
#             else:
#                 group = static_group

#             # Loop through variables in the .nc file
#             for var_name in ds.data_vars:
#                 print(f"  → Adding {var_name}")
#                 data = ds[var_name].values  # Load just this variable into memory
#                 group.create_dataset(var_name, data=data, compression='gzip')

#             ds.close()



# with h5py.File(out_file, "w", libver="latest") as h5file:
#         h5file.swmr_mode = True

#         # skip 'test' if there are no pretraining test paths
#         #  (ie if --test_fraction in preprocess.py was 0.0)
#         for split in ("ood", "val", "train"):
#             loader = dls[split]
#             logger.info("Running for split %s" % split)
#             ds = loader.dataset
#             size = len(ds) if not DEBUG else 10 * cfg.pretrain.batch_size
#             ds = h5file.create_dataset(
#                 split,
#                 (size, 7, TILE_SIZE, TILE_SIZE),
#                 chunks=(1, 7, TILE_SIZE, TILE_SIZE),
#                 dtype=np.float32,
#                 compression="gzip",
#             )
#             path_to_h5_idx[split] = {}

#             last = 0
#             for batch_idx, batch in enumerate(tqdm(iter(loader))):
#                 assert isinstance(batch.tiles, torch.Tensor)
#                 if DEBUG and batch_idx >= 10:
#                     break

#                 ds[last : last + batch.bs] = batch.tiles.squeeze(1).numpy()

#                 for i, tile in enumerate(batch.anchors):
#                     key = tile.path
#                     path_to_h5_idx[split][key] = last + i

#                 last += batch.bs

#         # https://docs.h5py.org/en/stable/vds.html
#         # https://github.com/h5py/h5py/blob/master/examples/dataset_concatenation.py
#         # create virtual dataset for downstream task, where train/val/test split is different (based on regions
#         # instead of on coordinates), so same downstream dataset (e.g. val) may need to access tiles from any
#         # pretraining split
#         nval = h5file["val"].shape[0]
#         ntrain = h5file["train"].shape[0]
#         nood = h5file["ood"].shape[0]
#         layout = h5py.VirtualLayout(
#             shape=(nval + ntrain + nood, 7, 145, 145), dtype=h5file["val"].dtype
#         )

#         valsrc = h5py.VirtualSource(h5file["val"])
#         trainsrc = h5py.VirtualSource(h5file["train"])
#         oodsrc = h5py.VirtualSource(h5file["ood"])

#         layout[:nval, :, :, :] = valsrc
#         layout[nval : nval + ntrain, :, :, :] = trainsrc
#         layout[nval + ntrain : nval + ntrain + nood, :, :, :] = oodsrc

#         h5file.create_virtual_dataset("all", layout)

#     path_to_virt_h5_idx = {
#         p: idx + (nval if s == "train" else (nval + ntrain if s == "ood" else 0))
#         for s in path_to_h5_idx
#         for (p, idx) in path_to_h5_idx[s].items()
#     }
#     with open("data/indices/path_to_h5_idx.json", "w") as f:
#         json.dump(path_to_h5_idx, f)
#     with open("data/indices/path_to_virtual_h5_idx.json", "w") as f:
#         json.dump(path_to_virt_h5_idx, f)