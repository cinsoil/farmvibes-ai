import shapely.geometry as shpg
from shapely.ops import transform as shapely_transform
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from vibe_core.client import get_default_vibe_client
from vibe_notebook.raster import read_raster, s2_to_img


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 4))
        color_mask = np.random.random(3)
        for i in range(3):
            img[:, :, i] = color_mask[i]
        img[:, :, 3] = 0.35
        ax.imshow(img * np.expand_dims(m, axis=2))


def get_sentinel2_raster(geometry, time_range, client):
    run = client.run(
        "data_ingestion/sentinel2/preprocess_s2",
        "Sentinel-2 download",
        geometry=geometry,
        time_range=time_range,
    )
    run.monitor()
    return run.output["raster"][0]


def load_and_show_image(raster_path, geometry):
    ar, transform = read_raster(raster_path, geometry)
    img = s2_to_img(ar)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    show(img.transpose((2, 0, 1)), transform=transform, ax=ax)
    ax.set_title("Sentinel-2 image")
    ax.set_axis_off()
    plt.show()

    return img, transform


def initialize_sam_model(model_name="vit_h", model_path="./models/sam_vit_h_4b8939.pth"):
    sam = sam_model_registry[model_name](checkpoint=model_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return sam


def generate_masks(sam_model, image):
    sam_input_img = (image * 255.0).astype(np.uint8)
    mask_generator = SamAutomaticMaskGenerator(sam_model, points_per_side=32)
    masks = mask_generator.generate(sam_input_img)
    return sam_input_img, masks


def plot_masks(image, masks):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.show()


def masks_to_polygons(masks, transform):
    polygons = []
    for mask in masks:
        m = mask["segmentation"]
        contours = plt.contour(m, levels=[0.5])
        for seg in contours.allsegs[0]:
            if len(seg) > 2:
                poly = Polygon(seg)
                poly = shapely_transform(lambda x, y: ~transform * (x, y), poly)
                polygons.append(poly)
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    return gdf


def export_to_geojson(gdf, filename="output_masks.geojson"):
    gdf.to_file(filename, driver="GeoJSON")
    print(f"Exported {len(gdf)} masks to {filename}")


def manual_prompt_segmentation(sam_model, image, input_point, input_label):
    predictor = SamPredictor(sam_model)
    predictor.set_image((image * 255).astype(np.uint8))
    masks, scores, logits = predictor.predict(
        point_coords=np.array([input_point]),
        point_labels=np.array([input_label]),
        multimask_output=True,
    )
    plt.figure(figsize=(10, 10))
    for i, mask in enumerate(masks):
        plt.subplot(1, len(masks), i + 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.title(f"Mask {i+1}")
        plt.axis("off")
    plt.show()
    return masks
def get_point_from_click(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Click on the object to segment (close window after click)")
    point = plt.ginput(1)[0]  # Get one point from user click
    plt.close()
    return [int(point[0]), int(point[1])]  # Return as integer pixel coords



def run_pipeline():
    geometry = shpg.Point(-119.21896203939313, 46.44578909859286).buffer(0.05, cap_style=3)
    time_range = (datetime(2020, 5, 1), datetime(2020, 5, 5))

    client = get_default_vibe_client()
    raster = get_sentinel2_raster(geometry, time_range, client)
    img, transform = load_and_show_image(raster, geometry)
    sam = initialize_sam_model()
    sam_input_img, masks = generate_masks(sam, img)
    plot_masks(sam_input_img, masks)

    gdf = masks_to_polygons(masks, transform)
    export_to_geojson(gdf)

    # Click on the image to select a point
    example_point = get_point_from_click(img)
    example_label = 1
    manual_prompt_segmentation(sam, img, example_point, example_label)

run_pipeline()
