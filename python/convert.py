import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
from PIL import Image
from pathlib import Path
import json  # To read the date for output filename

# --- Configuration ---
INPUT_GEOJSON = Path("data/seaice_geojson/_all/2000-01.json")  # Input file path
OUTPUT_DIR = Path("output_textures")  # Directory to save the PNG
RASTER_WIDTH = 2048  # pixels
RASTER_HEIGHT = 1024  # pixels
# Change interpolation method to 'linear' for smoother results
INTERPOLATION_METHOD = "linear"  # 'nearest', 'linear', 'cubic'
FILL_VALUE = 0.0  # Value for pixels outside data range (no ice)
# Define latitude threshold to exclude points very close to the poles
POLAR_LATITUDE_THRESHOLD = (
    89.0  # Degrees (exclude points above N or below S this latitude)
)

# --- Helper Functions ---


def create_raster_grid(width, height):
    """Creates longitude/latitude grids for the target raster dimensions."""
    # Create arrays of longitude and latitude values corresponding to pixel centers
    # Longitude ranges from -180 to +180 (exclusive of +180 endpoint for wrapping)
    # Latitude ranges from +90 to -90 (inclusive)
    lon_step = 360.0 / width
    lat_step = 180.0 / height
    # Pixel centers: start half a step in from the edge
    lon_centers = np.linspace(-180 + lon_step / 2, 180 - lon_step / 2, width)
    lat_centers = np.linspace(90 - lat_step / 2, -90 + lat_step / 2, height)
    # Create 2D grids
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    return lon_grid, lat_grid


def save_rgba_png(image_array, output_path):
    """Saves a NumPy RGBA array as a PNG image using Pillow."""
    try:
        # Ensure array is in uint8 format for Pillow
        if image_array.dtype != np.uint8:
            # Clip values just in case interpolation produced slightly out-of-bounds numbers
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        img = Image.fromarray(image_array, "RGBA")
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        img.save(output_path)
        print(f"Successfully saved image to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


# --- Main Processing ---
if __name__ == "__main__":
    print(f"Processing {INPUT_GEOJSON}...")

    # --- 1. Load GeoJSON Data ---
    try:
        gdf = gpd.read_file(INPUT_GEOJSON)
        # Read date for output filename
        with open(INPUT_GEOJSON, "r") as f:
            meta = json.load(f)
        date_str = meta.get("date", "unknown_date")  # e.g., "2000-01"
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_GEOJSON}")
        exit()
    except Exception as e:
        print(f"Error reading GeoJSON file {INPUT_GEOJSON}: {e}")
        exit()

    if gdf.empty:
        print("Error: GeoDataFrame is empty. No data to process.")
        exit()

    if "concentration" not in gdf.columns:
        print("Error: 'concentration' column not found in GeoJSON properties.")
        exit()

    # Extract points (lon, lat) and concentration values
    # GeoPandas geometry.x gives longitude, geometry.y gives latitude
    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y
    values = gdf["concentration"].values

    # --- Add Filtering Step ---
    # Filter out points too close to the poles to reduce projection artifacts
    original_point_count = len(gdf)
    gdf = gdf[gdf["latitude"].abs() <= POLAR_LATITUDE_THRESHOLD]
    filtered_point_count = len(gdf)
    print(
        f"Filtered out {original_point_count - filtered_point_count} points near poles (latitude > {POLAR_LATITUDE_THRESHOLD} or < -{POLAR_LATITUDE_THRESHOLD})."
    )

    # Prepare points and values for interpolation *after* filtering
    points = gdf[["longitude", "latitude"]].values
    values = gdf["concentration"].values

    # Filter out any potential NaN values in concentration if necessary
    valid_indices = ~np.isnan(values)
    points = points[valid_indices]
    values = values[valid_indices]

    if len(points) == 0:
        print("Error: No valid data points found after filtering.")
        exit()

    print(f"Using {len(points)} data points for interpolation.")

    # --- 2. Create Target Raster Grid ---
    print(f"Creating target grid ({RASTER_WIDTH}x{RASTER_HEIGHT})...")
    target_lon_grid, target_lat_grid = create_raster_grid(RASTER_WIDTH, RASTER_HEIGHT)

    # --- 3. Interpolate Data onto Grid ---
    # Using the updated INTERPOLATION_METHOD ('linear')
    print(f"Interpolating data using '{INTERPOLATION_METHOD}' method...")
    # griddata expects target points as a (N, 2) array, so stack lon/lat grids
    target_points = np.vstack((target_lon_grid.ravel(), target_lat_grid.ravel())).T

    # Perform interpolation
    interpolated_values = griddata(
        points,  # Input data coordinates (lon, lat)
        values,  # Input data values (concentration)
        target_points,  # Target grid coordinates
        method=INTERPOLATION_METHOD,
        fill_value=FILL_VALUE,
    )  # Value for points outside input data hull

    # Reshape the flat interpolated values back into the 2D grid shape
    interpolated_grid = interpolated_values.reshape((RASTER_HEIGHT, RASTER_WIDTH))
    print("Interpolation complete.")

    # --- 4. Create RGBA Image Array ---
    print("Creating RGBA image array...")
    # Initialize RGBA array (Height, Width, 4 channels) with zeros
    # All pixels start as black and fully transparent
    rgba_image = np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)

    # Where concentration > 0 (using a small epsilon to avoid floating point issues)
    # Use interpolated_grid which is now smoother
    ice_mask = interpolated_grid > 1e-6

    # Calculate grayscale intensity (0-255) based on concentration (0-1)
    # Clamp concentration values between 0 and 1 before scaling
    # Linear/cubic interpolation can sometimes yield values slightly outside 0-1
    concentration_clamped = np.clip(interpolated_grid[ice_mask], 0.0, 1.0)
    intensity = (concentration_clamped * 255).astype(np.uint8)

    # Set RGB channels to the intensity value for grayscale
    rgba_image[ice_mask, 0] = intensity  # Red
    rgba_image[ice_mask, 1] = intensity  # Green
    rgba_image[ice_mask, 2] = intensity  # Blue

    # Set Alpha channel to 255 (opaque) where there is ice
    rgba_image[ice_mask, 3] = 255

    # Pixels where ice_mask is False remain [0, 0, 0, 0] (transparent black)

    print("RGBA image array created.")

    # --- 5. Save Image ---
    # Add interpolation method and filter info to filename for clarity
    output_filename = f"sea_ice_{date_str}_{RASTER_WIDTH}x{RASTER_HEIGHT}_{INTERPOLATION_METHOD}_filt{POLAR_LATITUDE_THRESHOLD}.png"
    output_path = OUTPUT_DIR / output_filename
    save_rgba_png(rgba_image, output_path)

    print("Script finished.")
