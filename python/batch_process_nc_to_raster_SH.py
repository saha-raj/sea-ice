import xarray as xr
import rioxarray  # Extends xarray with .rio accessor - IMPORTANT to import
import numpy as np
from PIL import Image
from pathlib import Path
from rasterio.enums import Resampling  # For specifying resampling method
from affine import Affine
import re  # Import regular expressions for filename parsing
from scipy import ndimage  # Import scipy for dilation, labeling, and center of mass
from skimage import measure, morphology  # For contour finding and smoothing
import json  # For saving GeoJSON
import glob  # For finding all .nc files
import argparse  # For command line arguments
import datetime  # For date validation and comparison
import rasterio.transform # For coordinate transformation

# --- Configuration ---
INPUT_DIR = Path("data/sea-ice-SH-bremen")
# IMAGES_OUTPUT_DIR = Path("data/sea-ice-SH-images") # Commented out as image saving is disabled
OUTLINES_OUTPUT_DIR = Path("data/sea-ice-SH-outlines") # Updated output directory
RASTER_WIDTH = 2048
RASTER_HEIGHT = 1024
TARGET_CRS = "EPSG:4326"  # Target projection (lat/lon)

# Define colors for mapping (RGBA) - Still needed for threshold calculation
COLOR_ZERO_ICE = [0, 95, 153, 0]
COLOR_MAX_ICE = [255, 255, 255, 255]

# --- Configuration for Bordering ---
HIGH_CONC_THRESHOLD = 0.7  # Ice above this threshold triggers border generation
# --- Alpha Gradient for Border Layers (Fraction: 1.0 = opaque, 0.0 = transparent) ---
# Generate 3 steps fading from 1.0 down to 0.2
BORDER_ALPHA_VALUES = np.linspace(1.0, 0.1, 3)
DILATION_ITERATIONS = len(BORDER_ALPHA_VALUES)  # Set iterations based on alpha values

# --- Calculate Fixed RGB Color for Border Pixels ---
# Interpolate color based on HIGH_CONC_THRESHOLD using the NEW color range
c_zero_rgb = np.array(COLOR_ZERO_ICE[:3])
c_max_rgb = np.array(COLOR_MAX_ICE[:3])
border_color_rgb_float = c_zero_rgb + HIGH_CONC_THRESHOLD * (c_max_rgb - c_zero_rgb)
BORDER_COLOR_RGB = np.clip(border_color_rgb_float, 0, 255).astype(np.uint8)
print(f"Calculated fixed border color (RGB): {BORDER_COLOR_RGB}")

# --- Configuration for Pole Hole Filling ---
POLE_REGION_ROWS = RASTER_HEIGHT // 10
MIN_CONC_FOR_ICE = 1e-6

# --- NEW Configuration for SH Outline Generation ---
MIN_ICE_MASS_AREA = 500 # Minimum number of pixels for an ice mass to be included in the outline

# Function to extract date from filename
def extract_date_from_filename(filename):
    # Looking for pattern like asi-AMSR2-n6250-20150115-v5.4.nc
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        # Force day to 15 for output filename consistency
        # day = match.group(3)
        day = "15"
        return f"{year}-{month}-{day}", year, month
    return None, None, None

# --- Helper Function ---
# def save_rgba_png(image_array, output_path): # Commented out as image saving is disabled
#     """Saves a NumPy RGBA array as a PNG image using Pillow."""
#     try:
#         if image_array.dtype != np.uint8:
#             image_array = np.clip(image_array, 0, 255).astype(np.uint8)
#         img = Image.fromarray(image_array, "RGBA")
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         img.save(output_path)
#         print(f"Successfully saved image to {output_path}")
#     except Exception as e:
#         print(f"Error saving image to {output_path}: {e}")

# --- Main Processing Function ---
def process_netcdf_file(input_file, output_outline_path): # Removed output_image_path
    print(f"Processing {input_file}...")
    # Ensure output directory exists
    output_outline_path.parent.mkdir(parents=True, exist_ok=True)

    ds = None
    try:
        # --- 1. Open NetCDF and Select Data ---
        ds = xr.open_dataset(input_file, mask_and_scale=True, decode_coords="all")
        data_var_name = "z"
        if data_var_name not in ds:
            # Fallback for potential different variable names
            potential_vars = [v for v in ds.data_vars if len(ds[v].shape) >= 2]
            if not potential_vars:
                 raise ValueError("No suitable 2D+ data variables found in the NetCDF file.")
            data_var_name = potential_vars[0] # Take the first suitable variable
            print(f"Warning: Variable 'z' not found. Using variable '{data_var_name}' instead.")
            # raise ValueError(f"Variable '{data_var_name}' not found in the NetCDF file.")

        ice_conc_raw = ds[data_var_name]
        dims_to_squeeze = [
            dim
            for dim in ice_conc_raw.dims
            if dim not in ["x", "y"] and ice_conc_raw.sizes[dim] == 1
        ]
        if dims_to_squeeze:
            ice_conc_raw = ice_conc_raw.squeeze(dims_to_squeeze, drop=True)

        # --- 2. Scale Data ---
        print("Scaling data from percentage (0-100) to fraction (0-1)...")
        ice_conc_scaled = ice_conc_raw / 100.0

        # --- 3. Set CRS and Reproject ---
        ice_conc_scaled = ice_conc_scaled.rio.set_spatial_dims(
            x_dim="x", y_dim="y", inplace=True
        )
        if ice_conc_scaled.rio.crs is None:
            print("CRS not automatically detected, attempting to write EPSG:3411...")
            try:
                # Check for common grid mapping variable names
                grid_mapping_var = None
                if 'polar_stereographic' in ds:
                    grid_mapping_var = 'polar_stereographic'
                elif 'crs' in ds:
                     grid_mapping_var = 'crs'
                # Add other potential names if needed

                if grid_mapping_var and 'spatial_ref' in ds[grid_mapping_var].attrs:
                     ice_conc_scaled = ice_conc_scaled.rio.write_crs(
                         ds[grid_mapping_var].attrs["spatial_ref"], inplace=True
                     )
                     print(f"CRS written from '{grid_mapping_var}' attributes.")
                else:
                    print("Could not find standard grid mapping info, falling back to EPSG:3411.")
                    ice_conc_scaled = ice_conc_scaled.rio.write_crs("EPSG:3411", inplace=True)
            except Exception as e:
                print(f"Failed to write CRS, falling back to EPSG:3411. Error: {e}")
                ice_conc_scaled = ice_conc_scaled.rio.write_crs("EPSG:3411", inplace=True)

        if "x" not in ice_conc_scaled.dims or "y" not in ice_conc_scaled.dims:
            raise ValueError("Spatial dimensions 'x' and 'y' not found after setting.")

        print(f"Source CRS: {ice_conc_scaled.rio.crs}")
        print(f"Reprojecting to {TARGET_CRS} ({RASTER_WIDTH}x{RASTER_HEIGHT})...")

        lon_res = 360.0 / RASTER_WIDTH
        lat_res = 180.0 / RASTER_HEIGHT
        target_transform = Affine(lon_res, 0.0, -180.0, 0.0, -lat_res, 90.0)

        ice_conc_reprojected = ice_conc_scaled.rio.reproject(
            dst_crs=TARGET_CRS,
            shape=(RASTER_HEIGHT, RASTER_WIDTH),
            transform=target_transform,
            resampling=Resampling.bilinear,
            nodata=np.nan,
        )
        print("Reprojection complete.")
        final_grid = ice_conc_reprojected.values
        # Get the affine transform of the *reprojected* grid for coordinate conversion
        final_transform = ice_conc_reprojected.rio.transform()

        # --- Step 3a: Fill Pole Hole Artifact (Optional but recommended for SH) ---
        # (Keep the existing pole hole filling logic as it might still be relevant)
        print("Attempting to fill pole hole artifact...")
        grid_with_hole_filled = final_grid.copy()
        pole_hole_filled = False
        try:
            # Create mask of non-ice pixels (NaN or near-zero) in the pole region
            # For SH, check the BOTTOM rows
            is_non_ice = np.isnan(final_grid[-POLE_REGION_ROWS:, :]) | (final_grid[-POLE_REGION_ROWS:, :] <= MIN_CONC_FOR_ICE)

            # Label connected components of non-ice in the pole region
            structure = ndimage.generate_binary_structure(2, 1) # 4-connectivity
            labeled_array, num_features = ndimage.label(is_non_ice, structure=structure)

            if num_features > 0:
                print(f"  Found {num_features} non-ice component(s) in the pole region.")
                # Calculate center of mass for each component (relative to the bottom region slice)
                centers = ndimage.center_of_mass(is_non_ice, labeled_array, range(1, num_features + 1))
                # Target center: bottom row, middle column (relative to the slice)
                target_center = (POLE_REGION_ROWS - 1, RASTER_WIDTH // 2)

                closest_feature_label = -1
                min_dist_sq = float('inf')

                if num_features == 1:
                    centers = [centers] # Make it a list

                for i, center in enumerate(centers):
                    label = i + 1
                    dist_sq = (center[0] - target_center[0])**2 + (center[1] - target_center[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_feature_label = label

                if closest_feature_label != -1:
                    print(f"  Identified component {closest_feature_label} as potential pole hole.")
                    # Create mask for the identified pole hole component (only in the bottom rows)
                    pole_hole_mask_region = (labeled_array == closest_feature_label)
                    # Create a full-sized mask for dilation
                    pole_hole_mask_full = np.zeros_like(final_grid, dtype=bool)
                    pole_hole_mask_full[-POLE_REGION_ROWS:, :] = pole_hole_mask_region # Apply to bottom rows

                    # Dilate the hole mask by 1 pixel to find the boundary
                    dilated_hole_mask = ndimage.binary_dilation(pole_hole_mask_full, structure=structure, iterations=1)
                    # Boundary pixels are in the dilated mask but not the original hole mask
                    hole_boundary_mask = dilated_hole_mask & ~pole_hole_mask_full

                    # Get coordinates of boundary pixels and their values in the original grid
                    boundary_coords = np.argwhere(hole_boundary_mask)
                    boundary_values = final_grid[hole_boundary_mask]

                    # Filter out NaN boundary values (shouldn't happen if nodata is handled, but safe)
                    valid_boundary_values = boundary_values[~np.isnan(boundary_values)]

                    if valid_boundary_values.size > 0:
                        # Calculate the average concentration of the valid boundary pixels
                        fill_value = np.mean(valid_boundary_values)
                        print(f"  Filling hole with average boundary value: {fill_value:.4f}")
                        # Fill the original hole mask region in the grid copy
                        grid_with_hole_filled[pole_hole_mask_full] = fill_value
                        pole_hole_filled = True
                    else:
                        print("  Warning: No valid boundary pixels found to calculate fill value.")
                else:
                     print("  Could not identify a central pole hole component.")
            else:
                print("  No non-ice components found in the pole region.")

        except Exception as e:
            print(f"  Error during pole hole filling: {e}")

        # Use the filled grid if successful, otherwise use the original
        if pole_hole_filled:
            final_grid_for_outline = grid_with_hole_filled
            print("Using grid with pole hole filled for outline generation.")
        else:
            final_grid_for_outline = final_grid
            print("Using original grid for outline generation (pole hole not filled or not found).")


        # --- 4. Generate RGBA Image (COMMENTED OUT) ---
        # print("Generating RGBA image...")
        # # Create an RGBA image array (Height, Width, 4 channels)
        # rgba_image = np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)
        #
        # # Create masks for different conditions
        # nan_mask = np.isnan(final_grid)
        # zero_ice_mask = (~nan_mask) & (final_grid <= MIN_CONC_FOR_ICE) # Treat near-zero as zero
        # ice_mask = (~nan_mask) & (final_grid > MIN_CONC_FOR_ICE)
        #
        # # Apply colors:
        # # Transparent blueish for zero/near-zero ice
        # rgba_image[zero_ice_mask] = COLOR_ZERO_ICE
        #
        # # Interpolate color and alpha for ice areas
        # if np.any(ice_mask):
        #     ice_values = final_grid[ice_mask] # Get concentration values (0 to 1)
        #
        #     # Interpolate RGB channels
        #     for i in range(3): # R, G, B
        #         rgba_image[ice_mask, i] = np.interp(
        #             ice_values,
        #             [0.0, 1.0], # Input range (concentration)
        #             [COLOR_ZERO_ICE[i], COLOR_MAX_ICE[i]] # Output range (color channel)
        #         ).astype(np.uint8)
        #
        #     # Interpolate Alpha channel (from transparent blue alpha to opaque white alpha)
        #     rgba_image[ice_mask, 3] = np.interp(
        #         ice_values,
        #         [0.0, 1.0],
        #         [COLOR_ZERO_ICE[3], COLOR_MAX_ICE[3]]
        #     ).astype(np.uint8)
        #
        # # --- Apply Bordering ---
        # print("Applying bordering effect...")
        # # Create initial high concentration mask using the grid potentially with hole filled
        # high_conc_mask_border = final_grid_for_outline >= HIGH_CONC_THRESHOLD
        # dilated_layers = [high_conc_mask_border.copy()] # Start with the core high-conc area
        #
        # # Perform successive dilations
        # structure = ndimage.generate_binary_structure(2, 1) # 4-connectivity is usually fine
        # for _ in range(DILATION_ITERATIONS):
        #     dilated_layers.append(ndimage.binary_dilation(dilated_layers[-1], structure=structure, border_value=0))
        #
        # # Apply colors layer by layer, from outermost to innermost
        # for i in range(DILATION_ITERATIONS, 0, -1):
        #     # Pixels in this layer are those in dilated_layers[i] but not in dilated_layers[i-1]
        #     layer_mask = dilated_layers[i] & ~dilated_layers[i-1]
        #     alpha_value = int(BORDER_ALPHA_VALUES[i-1] * 255) # Get alpha for this layer
        #     # Apply fixed RGB color and calculated alpha
        #     rgba_image[layer_mask, 0] = BORDER_COLOR_RGB[0]
        #     rgba_image[layer_mask, 1] = BORDER_COLOR_RGB[1]
        #     rgba_image[layer_mask, 2] = BORDER_COLOR_RGB[2]
        #     rgba_image[layer_mask, 3] = alpha_value
        #
        # # Ensure the core high concentration area (innermost layer) is fully opaque white
        # core_mask = dilated_layers[0] # This is the original high_conc_mask_border
        # rgba_image[core_mask] = COLOR_MAX_ICE # Set core to max color (opaque white)
        #
        # # --- Save Image (COMMENTED OUT) ---
        # # save_rgba_png(rgba_image, output_image_path)
        # print("Image saving skipped.")

        # --- 5. Generate Outlines for All Sufficiently Large Ice Masses ---
        print(f"Generating outlines for components >= {MIN_ICE_MASS_AREA} pixels using threshold: {HIGH_CONC_THRESHOLD}...")
        # Create a binary mask based on the high concentration threshold
        # Handle potential NaN values by treating them as below threshold
        ice_mask = ~np.isnan(final_grid) & (final_grid >= HIGH_CONC_THRESHOLD)

        # Label connected components (ice masses) in the mask
        # Use 4-connectivity (structure=ndimage.generate_binary_structure(2, 1))
        structure = ndimage.generate_binary_structure(2, 1)
        labeled_array, num_features = ndimage.label(ice_mask, structure=structure)
        print(f"  Found {num_features} potential ice component(s).")

        all_geojson_features = [] # List to hold features for the FeatureCollection

        if num_features > 0:
            # Calculate the size of each labeled component
            component_sizes = np.bincount(labeled_array.ravel())
            # component_sizes[0] is the background, ignore it

            for i in range(1, num_features + 1): # Iterate through each component label
                area = component_sizes[i]
                if area >= MIN_ICE_MASS_AREA:
                    print(f"  Processing component {i} (area: {area} pixels)...")
                    # Create a mask for only the current component
                    component_mask = (labeled_array == i)

                    # Find contours for this specific component's boundary
                    # Use level 0.5 to find edge between False (0) and True (1)
                    contours = measure.find_contours(component_mask.astype(float), 0.5)

                    if not contours:
                        print(f"    Warning: No contour found for component {i} despite area threshold.")
                        continue

                    # Usually, the longest contour is the exterior boundary
                    component_contour = max(contours, key=len)
                    print(f"    Found contour with {len(component_contour)} points.")

                    # Smooth the contour
                    smoothed_contour = smooth_contour(component_contour, tolerance=1.0)
                    print(f"    Smoothed contour has {len(smoothed_contour)} points.")

                    # Convert contour coordinates (row, col) to geographic coordinates (lon, lat)
                    cols = smoothed_contour[:, 1]
                    rows = smoothed_contour[:, 0]
                    lons, lats = rasterio.transform.xy(final_transform, rows, cols)

                    # Format coordinates as [lon, lat] pairs for GeoJSON LineString
                    contour_geo = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]

                    # Create a GeoJSON Feature for this component's outline
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "area_pixels": int(area),
                            "original_points": len(component_contour),
                            "smoothed_points": len(smoothed_contour)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": contour_geo
                        }
                    }
                    all_geojson_features.append(feature)
                # else: # Optional: Log skipped small components
                    # print(f"  Skipping component {i} (area: {area} pixels) - below threshold.")


        # --- 6. Save Outlines as GeoJSON FeatureCollection ---
        if all_geojson_features:
            # Create the GeoJSON FeatureCollection structure
            geojson_data = {
                "type": "FeatureCollection",
                "features": all_geojson_features
            }
            try:
                output_outline_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                with open(output_outline_path, 'w') as f:
                    # Use separators for compact output
                    json.dump(geojson_data, f, separators=(',', ':'))
                print(f"Successfully saved {len(all_geojson_features)} outline(s) to {output_outline_path}")
            except Exception as e:
                print(f"!!! Error saving GeoJSON FeatureCollection to {output_outline_path}: {e}")
                # --- ADD TRACEBACK ---
                import traceback
                traceback.print_exc() # Print the full traceback for detailed debugging
                # --- END ADD ---
        else:
            print(f"No ice components met the minimum area threshold. No outline saved to {output_outline_path}.")


    except Exception as e:
        print(f"!!! ERROR processing {input_file}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure the dataset is closed
        if ds:
            ds.close()

# --- Helper function for contour smoothing ---
def smooth_contour(contour, tolerance=1.0):
    """Smooths a contour using the Ramer-Douglas-Peucker algorithm."""
    # approximate_polygon is faster for simple smoothing
    return measure.approximate_polygon(contour, tolerance=tolerance)


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process NetCDF sea ice data to GeoJSON outlines for SH.")
    parser.add_argument(
        "--input_dir", type=Path, default=INPUT_DIR, help="Directory containing input NetCDF files."
    )
    # parser.add_argument( # Image output disabled
    #     "--img_output_dir", type=Path, default=IMAGES_OUTPUT_DIR, help="Directory to save output PNG images."
    # )
    parser.add_argument(
        "--outline_output_dir", type=Path, default=OUTLINES_OUTPUT_DIR, help="Directory to save output GeoJSON outlines."
    )
    parser.add_argument(
        "--start_date", type=str, default="1970-01-01", help="Start date (YYYY-MM-DD). Process files on or after this date."
    )
    parser.add_argument(
        "--end_date", type=str, default="2100-12-31", help="End date (YYYY-MM-DD). Process files on or before this date."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )

    args = parser.parse_args()

    # Validate dates
    try:
        start_dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        exit(1)

    # Find all NetCDF files
    nc_files = sorted(list(args.input_dir.glob("*.nc")))
    print(f"Found {len(nc_files)} NetCDF files in {args.input_dir}.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for nc_file in nc_files:
        date_str, year, month = extract_date_from_filename(nc_file.name)
        if date_str:
            try:
                file_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                # Check if file date is within the specified range
                if not (start_dt <= file_dt <= end_dt):
                    # print(f"Skipping {nc_file.name} (date {date_str} outside range {args.start_date} to {args.end_date}).")
                    skipped_count += 1
                    continue

                # Construct output paths using the forced '15' day from extract_date_from_filename
                # output_image_path = args.img_output_dir / f"{date_str}.png" # Image output disabled
                output_outline_path = args.outline_output_dir / f"{date_str}.geojson"

                # Check if output exists and skip if overwrite is not enabled
                if not args.overwrite and output_outline_path.exists():
                    print(f"Skipping {nc_file.name} (outline already exists at {output_outline_path}).")
                    skipped_count += 1
                    continue

                # Process the file (pass only outline path)
                process_netcdf_file(nc_file, output_outline_path)
                processed_count += 1

            except Exception as e:
                print(f"!!! FAILED processing {nc_file.name}: {e}")
                error_count += 1
        else:
            print(f"Skipping file with unrecognized date format: {nc_file.name}")
            skipped_count += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (out of date range or existing): {skipped_count}")
    print(f"Errors: {error_count}")
    print("------------------------") 