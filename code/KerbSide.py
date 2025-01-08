import os
import json
import requests
from time import sleep
from geopy.geocoders import Nominatim
import osm2streets_python
import geopandas as gpd
import pandas as pd
import argparse
import math
import xml.etree.ElementTree as ET

GLOBAL_CACHE_FILE = "../data/raw_data/osm2streets/location_cache.json"

def get_location_info(geolocator, location_name):
    """
    Retrieve location information, including bounding box and OSMID, for a given location name.

    Args:
        geolocator (geopy.geocoders.Nominatim): Geolocator instance.
        location_name (str): Name of the location to query.

    Returns:
        tuple: OSMID (int) and bounding box (tuple of floats).

    Raises:
        ValueError: If the location or OSMID cannot be found.
    """
    # Load the global cache
    cache = load_location_cache()

    # Check if location is already cached
    if location_name in cache:
        print(f"Using cached data for {location_name}")
        osmid = cache[location_name]["osmid"]
        bbox = tuple(cache[location_name]["boundingbox"])
        return osmid, bbox

    # Query Nominatim if not cached
    print(f"Querying Nominatim for {location_name}...")
    location = geolocator.geocode(location_name)
    if location:
        osmid = location.raw.get("osm_id", None)
        if not osmid:
            raise ValueError("OSMID not found for location.")
        bbox = (
            float(location.raw['boundingbox'][0]),
            float(location.raw['boundingbox'][1]),
            float(location.raw['boundingbox'][2]),
            float(location.raw['boundingbox'][3]),
        )

        # Save to the global cache
        cache[location_name] = {"osmid": osmid, "boundingbox": bbox}
        save_location_cache(cache)
        return osmid, bbox
    else:
        raise ValueError(f"Location not found: {location_name}")
    
def load_location_cache():
    """
    Load the location cache from a global file.

    Returns:
        dict: A dictionary with cached location data.
    """
    if os.path.exists(GLOBAL_CACHE_FILE):
        with open(GLOBAL_CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_location_cache(cache):
    """
    Save the location cache to a global file.

    Args:
        cache (dict): The location cache dictionary to save.
    """
    with open(GLOBAL_CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

def initialize_geolocator(user_agent="osm2streets_python/0.1.0"):
    """
    Initialize the geolocator for querying location data.
    
    Args:
        user_agent (str): User agent string for the Nominatim geolocator.

    Returns:
        geopy.geocoders.Nominatim: Initialized geolocator instance.
    """
    return Nominatim(user_agent=user_agent)

# TODO: Delete or deal differently with _fixed files if they are there already. 
# Currently these files are created multiple times
def list_non_hidden_files(directory):
    """
    List non-hidden files in a directory, excluding files ending with '_fixed.osm'.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of non-hidden files excluding '_fixed.osm' files.
    """
    return [f for f in os.listdir(directory) if not f.startswith('.') and not f.endswith('_fixed.osm')]

def download_tiles(osmid, bbox, tile_size, output_dir, overpass_url="https://lz4.overpass-api.de/api/interpreter"):
    """
    Download OSM tiles within the specified bounding box, divided into smaller tiles.

    Args:
        osmid (int): OpenStreetMap ID for the location.
        bbox (tuple): Bounding box (min_lat, max_lat, min_lon, max_lon).
        tile_size (float): Size of each tile in degrees.
        output_dir (str): Directory to save the downloaded tiles.
        overpass_url (str): URL of the Overpass API.

        Alternative overpass_urls: 
        https://z.overpass-api.de/api/interpreter
        http://overpass-api.de/api/interpreter

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    min_lat, max_lat, min_lon, max_lon = bbox
    lat_steps = int((max_lat - min_lat) / tile_size) + 1
    lon_steps = int((max_lon - min_lon) / tile_size) + 1
    print(f"Creating {lat_steps * lon_steps} tiles...")

    print(f"Found {len(list_non_hidden_files(output_dir))} files in output folder")

    if len(list_non_hidden_files(output_dir)) != (lat_steps * lon_steps):
        tile_count = 0
        for i in range(lat_steps):
            for j in range(lon_steps):
                tile_min_lat = min_lat + i * tile_size
                tile_max_lat = min(tile_min_lat + tile_size, max_lat)
                tile_min_lon = min_lon + j * tile_size
                tile_max_lon = min(tile_min_lon + tile_size, max_lon)

                bbox = (tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon)
                ways_and_nodes_query = f"""
                [out:xml];
                (
                way["highway"]["area"!~"yes"]["highway"!~"abandoned|construction|no|planned|platform|proposed|raceway|razed"]
                ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
                >;);
                out;
                """

                sleep(1)  # Add delay between requests
                headers = {"User-Agent": "osm2streets_python/0.1.0"}
                response = requests.post(overpass_url, data=ways_and_nodes_query, headers=headers)
                if response.status_code == 200:
                    print(f"Tile {tile_count + 1}/{lat_steps * lon_steps} fetched successfully!")
                    tile_file = os.path.join(output_dir, f"{osmid}_tile_{tile_count}.osm")
                    with open(tile_file, "w") as f:
                        f.write(response.text)
                    tile_count += 1
                else:
                    print(f"Error fetching tile {tile_count + 1}: {response.status_code}")
                sleep(1)  # Add delay between requests
        print(f"Downloaded {tile_count} tiles to {output_dir}.")
    else:
        print(f"Files for {osmid} are already tiled and downloaded. Skipping download for now")



def has_repeat_non_adjacent_points(coords):
    """
    Check if a sequence of coordinates contains repeat non-adjacent points.

    Args:
        coords (list): List of (lat, lon) tuples.

    Returns:
        bool: True if repeat non-adjacent points are found, False otherwise.
    """
    seen = set()
    for i, coord in enumerate(coords):
        if coord in seen and coord != coords[i - 1]:  # Allow adjacent repeats
            return True
        seen.add(coord)
    return False


def find_problematic_ways(osm_file, epsilon_dist=1e-5):
    """
    Identify ways with repeat non-adjacent points in an OSM file.

    Args:
        osm_file (str): Path to the OSM file.
        epsilon_dist (float): Minimum allowed length for a line segment.

    Returns:
        list: IDs of problematic ways.
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()
    nodes = {}  # Dictionary to map node IDs to their coordinates
    
    
    # Extract node coordinates
    for node in root.findall("node"):
        node_id = node.get("id")
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lat, lon)

    # Check ways for repeat non-adjacent points
    problematic_ways = []
    for way in root.findall("way"):
        way_id = way.get("id")
        coords = [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]
        if has_repeat_non_adjacent_points(coords):
            print(f"Way {way_id} has repeat non-adjacent points.")
            problematic_ways.append(way_id)
        
        # Check for degenerate lines (length < epsilon_dist)
        for i in range(len(coords) - 1):
            pt1, pt2 = coords[i], coords[i + 1]
            dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if dist < epsilon_dist:
                print(f"Problematic way {way_id}: Degenerate segment from {pt1} to {pt2}")
                problematic_ways.append(way_id)
                break

    return problematic_ways

def is_valid_osm_file(osm_input):
    """
    Validate if the .osm file contains essential OSM data elements.

    Args:
        osm_input (bytes): Raw .osm file content.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    try:
        content = osm_input.decode("utf-8")
        # Check for essential OSM elements (node, way, relation)
        if any(tag in content for tag in ["<node", "<way", "<relation"]):
            return True
    except Exception as e:
        print(f"Error validating .osm file: {e}")
    return False



def fix_or_remove_invalid_ways(osm_file, output_file):
    """
    Fix or remove ways with repeat non-adjacent points in an OSM file.

    Args:
        osm_file (str): Path to the input OSM file.
        output_file (str): Path to save the fixed OSM file.

    Returns:
        list: IDs of removed ways.
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()
    nodes = {}  # Dictionary to map node IDs to their coordinates

    # Extract node coordinates
    for node in root.findall("node"):
        node_id = node.get("id")
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[node_id] = (lat, lon)

    # Check and fix ways
    removed_ways = []
    for way in root.findall("way"):
        way_id = way.get("id")
        coords = [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]

        # Check for repeat non-adjacent points
        seen = set()
        has_problem = False
        fixed_coords = []
        for i, coord in enumerate(coords):
            if coord in seen and coord != coords[i - 1]:  # Allow adjacent repeats
                has_problem = True
            else:
                fixed_coords.append(coord)
            seen.add(coord)

        if has_problem:
            if len(fixed_coords) >= 2:  # Retain only if valid polyline remains
                print(f"Fixing way {way_id} by removing problematic points.")
                # Update the way with fixed coordinates
                for nd, coord in zip(way.findall("nd"), fixed_coords):
                    node_id = [key for key, value in nodes.items() if value == coord][0]
                    nd.set("ref", node_id)
            else:
                print(f"Removing way {way_id} due to invalid geometry.")
                root.remove(way)
                removed_ways.append(way_id)

    # Save the fixed OSM file
    tree.write(output_file)
    return removed_ways

def filter_problematic_ways(osm_file, output_file, problematic_ways):
    """
    Filter out problematic ways from an OSM file and save the result.

    Args:
        osm_file (str): Path to the original OSM file.
        output_file (str): Path to save the filtered OSM file.
        problematic_ways (list): List of IDs of problematic ways.

    Returns:
        None
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Remove problematic ways
    for way in root.findall("way"):
        if way.get("id") in problematic_ways:
            root.remove(way)

    # Save the filtered file
    tree.write(output_file)
    print(f"Filtered OSM file saved to {output_file}")


def add_xml_header_if_missing(osm_file_path):
    '''
    If there's no osm data in the current tile, it creates a invalid osm file without a header. This function adds the proper header if missing so it doesn't prompt errors in further processing

    Args:
        osm_file_path (str): Path to the OSM file

    Returns:
        None
    '''
    with open(osm_file_path, 'r+') as file:
        content = file.read()
        if not content.startswith('<?xml version="1.0" encoding="UTF-8"?>'):
            file.seek(0, 0)  # Move to the start of the file
            file.write('<?xml version="1.0" encoding="UTF-8"?>\n' + content)

def process_tiles(tile_dir, input_options, output_dir, output_options=None):
    '''
        Processes OSM tile files from a specified directory, applying fixes to problematic ways,
    extracting GeoDataFrames of street networks, lanes, and intersections, and saving the results.

    Parameters:
    -----------
    tile_dir : str
        The directory containing OSM tile files to be processed.
    input_options : dict
        A dictionary of options for processing OSM data using the osm2streets library. This is 
        converted to a JSON string for compatibility with the library.
    output_dir : str
        The directory where the processed GeoJSON files and logs will be saved. The directory will
        be created if it doesn't exist.
        
    output_options : dict, optional
        A dictionary specifying which outputs to generate. Possible keys:
            - "network": bool, default True
                Whether to output the plain street network geometry.
            - "lanes": bool, default True
                Whether to output lane polygons.
            - "intersections": bool, default True
                Whether to output intersection markings.

    Outputs:
    --------
    Depending on `output_options`, the function generates:
        - `combined_network.geojson`
        - `combined_lanes.geojson`
        - `combined_intersections.geojson`
    A log file (`failed_tiles.txt`) is always generated for tiles that failed to process.

    Process Overview:
    -----------------
    1. Iterates through all `.osm` files in the specified `tile_dir`.
    2. Detects and fixes problematic ways within each tile, if necessary.
    3. Processes each tile using the `osm2streets` Python bindings to extract GeoDataFrames for:
        - Plain street network geometry.
        - Lane polygons.
        - Intersection markings (if implemented).
    4. Concatenates the GeoDataFrames from all tiles into combined GeoDataFrames.
    5. Saves the combined GeoDataFrames as GeoJSON files in the `output_dir`.
    6. Logs any tiles that failed to process.

    Steps in Detail:
    ----------------
    1. **Input Directory Validation**:
        - All `.osm` files in the `tile_dir` are identified for processing.
    2. **Fix Problematic Ways**:
        - Calls `find_problematic_ways` to identify invalid or problematic ways.
        - Attempts to fix or remove these ways using `fix_or_remove_invalid_ways`.
        - Ensures the corrected tile file has a valid XML header using `add_xml_header_if_missing`.
    3. **OSM File Processing**:
        - Reads the OSM tile file as binary data and initializes a `PyStreetNetwork` object.
        - Extracts GeoDataFrames for network geometry, lanes, and (optionally) intersections.
    4. **Error Handling**:
        - Catches and logs any errors encountered during processing.
    5. **Output Generation**:
        - Saves combined GeoDataFrames to GeoJSON files.
        - Creates a log file for tiles that failed to process.

    Exceptions:
    -----------
    - Any unhandled errors during tile processing are caught and logged, ensuring that processing continues for other tiles.
    '''
    # Set default output options if none are provided
    if output_options is None:
        output_options = {"network": True, "lanes": True, "intersections": True}

    os.makedirs(output_dir, exist_ok=True)

    # Initialize combined GeoDataFrames for selected outputs
    combined_gdf = gpd.GeoDataFrame() if output_options.get("network", True) else None
    combined_gdf_lanes = gpd.GeoDataFrame() if output_options.get("lanes", True) else None
    combined_gdf_intersections = gpd.GeoDataFrame() if output_options.get("intersections", True) else None

    failed_tiles = []

    for tile_file in os.listdir(tile_dir):
        if tile_file.endswith(".osm"):
            tile_path = os.path.join(tile_dir, tile_file)
            print(f"Processing tile: {tile_file}")

            # Detect and fix problematic ways
            problematic_ways = find_problematic_ways(tile_path)
            if problematic_ways:
                print(f"Fixing problematic ways in {tile_file}: {problematic_ways}")
                fixed_tile_path = tile_path.replace(".osm", "_fixed.osm")
                removed_ways = fix_or_remove_invalid_ways(tile_path, fixed_tile_path)
                add_xml_header_if_missing(fixed_tile_path)
                print(f"Removed ways: {removed_ways}")
                tile_path = fixed_tile_path

            # Process the (potentially fixed) tile
            try:
                with open(tile_path, "rb") as file:
                    osm_input = file.read()

                network = osm2streets_python.PyStreetNetwork(osm_input, "", json.dumps(input_options))

                # Extract GeoDataFrames based on output options
                if output_options.get("network", True):
                    gdf = validate_geometry(network.to_geojson_plain)
                    if not gdf.empty:
                        combined_gdf = gpd.GeoDataFrame(pd.concat([combined_gdf, gdf], ignore_index=True))

                if output_options.get("lanes", True):
                    gdf_lanes = validate_geometry(network.to_lane_polygons_geojson)
                    if not gdf_lanes.empty:
                        combined_gdf_lanes = gpd.GeoDataFrame(pd.concat([combined_gdf_lanes, gdf_lanes], ignore_index=True))

                if output_options.get("intersections", True):
                    gdf_intersections = validate_geometry(network.to_intersection_markings_geojson)
                    if not gdf_intersections.empty:
                        combined_gdf_intersections = gpd.GeoDataFrame(pd.concat([combined_gdf_intersections, gdf_intersections], ignore_index=True))

            except Exception as e:
                print(f"Unhandled error processing tile {tile_file}: {e}")
                failed_tiles.append(tile_file)
                continue

    # Save selected outputs
    if output_options.get("network", True) and combined_gdf is not None:
        combined_gdf.to_file(os.path.join(output_dir, "combined_network.geojson"), driver="GeoJSON")
    if output_options.get("lanes", True) and combined_gdf_lanes is not None:
        combined_gdf_lanes.to_file(os.path.join(output_dir, "combined_lanes.geojson"), driver="GeoJSON")
    if output_options.get("intersections", True) and combined_gdf_intersections is not None:
        combined_gdf_intersections.to_file(os.path.join(output_dir, "combined_intersections.geojson"), driver="GeoJSON")

    # Log failed tiles
    if failed_tiles:
        print(f"Failed tiles: {failed_tiles}")
        with open(os.path.join(output_dir, "failed_tiles.txt"), "w") as log_file:
            log_file.write("\n".join(failed_tiles))


def validate_geometry(geojson_func):
    """
    Safely execute a function that generates GeoJSON and validate the resulting geometry.

    Args:
        geojson_func (callable): Function to generate GeoJSON.

    Returns:
        geopandas.GeoDataFrame: Validated GeoDataFrame, or an empty GeoDataFrame on failure.
    """
    try:
        geojson_data = geojson_func()
        gdf = gpd.GeoDataFrame.from_features(json.loads(geojson_data)["features"])
        # Ensure GeoDataFrame contains valid geometries
        if not gdf.empty and "geometry" in gdf.columns:
            gdf = gdf[gdf.is_valid]  # Drop invalid geometries
            gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    except Exception as e:
        print(f"Geometry validation failed: {e}")
        return gpd.GeoDataFrame()


def main(location_name, tile_size, driving_side, output_options):
    """
    Main function to orchestrate tile downloading and processing for a given location.

    Args:
        location_name (str): Name of the location to process.
        tile_size (float): Size of each tile in degrees.
        driving_side (str): Driving side ('Right' or 'Left').

    Returns:
        None
    """
    os.environ["RUST_LOG"] = "off"
    os.environ["RUST_BACKTRACE"] = "full"
    geolocator = initialize_geolocator()

    try:
        osmid, bbox = get_location_info(geolocator, location_name)
        base_dir = f"../data/raw_data/osm2streets/{osmid}"
        tile_dir = os.path.join(base_dir, "tiles")
        processed_dir = os.path.join(base_dir, "processed")

        # Step 1: Download Tiles
        download_tiles(osmid, bbox, tile_size, tile_dir)

        # Step 2: Process Tiles
        input_options = {
            "debug_each_step": False,
            "dual_carriageway_experiment": False,
            "sidepath_zipping_experiment": False,
            "inferred_sidewalks": True,
            "inferred_kerbs": True,
            "date_time": None,
            "override_driving_side": driving_side
        }
        process_tiles(tile_dir, input_options, processed_dir, output_options)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OSM data for a given location.")
    parser.add_argument("location", type=str, help="Name of the location to process (e.g., 'West, Amsterdam').")
    parser.add_argument("--tile_size", type=float, default=0.01, help="Tile size in degrees (default: 0.01).")
    parser.add_argument("--driving_side", type=str, default="Right", choices=["Right", "Left"],
                        help="Driving side ('Right' or 'Left', default: 'Right').")
    parser.add_argument("--outputs", type=dict, default=None, help="Outputs as dict: {'network': True, 'lanes': True, 'intersections': True}, default=all")
    args = parser.parse_args()
    main(args.location, args.tile_size, args.driving_side, args.outputs)