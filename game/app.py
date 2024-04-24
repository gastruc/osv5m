"""Requires gradio==3.44.0"""
import io
import shutil 
import os
import uuid
import matplotlib
import time
import pathlib

from PIL import Image
from math import radians, sin, cos, sqrt, asin, exp
from os.path import join
from collections import defaultdict

import matplotlib.style as mplstyle
mplstyle.use(['fast'])
import pandas as pd

import gradio as gr
import wandb
import reverse_geocoder as rg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from gradio_folium import Folium
from folium import Map, Element, LatLngPopup
from matplotlib.offsetbox import AnchoredText


IMAGE_FOLDER = './select'
CSV_FILE = './select.csv'
RESULTS_DIR = './results'
RULES = """<h1>OSV-5M (plonk)</h1>
<center><img width="256" alt="Rotating globe" src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Rotating_globe.gif"></center>
<h2> Instructions </h2>
<h3> Click on the map 🗺️ (left) to the location at which you think the image 🖼️ (right) was captured! </h3>
<h3>⚠️ Your selection is final!</h3>
<h3> Click next to move to the next image. </h3>
"""
css = """
@font-face {
  font-family: custom;
  src: url("/file=custom.ttf");
}

h1 {
    text-align: center;
    display:block;
    font-family: custom;
}
img {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
    font-family: custom;
}
h3 {
    text-align: center;
    display:block;
    font-family: custom;
    font-weight: normal;
}
"""

space_js = """
<script>
function shortcuts(e) {
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        break;
        default:
        if (e.key.toLowerCase() == " " && !e.shiftKey) {
            document.getElementById("latlon_btn").click();
        }
    }
}
document.addEventListener('keypress', shortcuts, false);
</script>
"""

def inject_javascript(folium_map):
    js = """
    document.addEventListener('DOMContentLoaded', function() {
        map_name_1.on('click', function(e) {
            window.state_data = e.latlng
        });
    });
    """
    folium_map.get_root().html.add_child(Element(f'<script>{js}</script>'))

def make_map_(name="map_name", id="1"):
    map = Map(location=[39, 23], zoom_start=1)
    map._name, map._id = "map_name", "1"

    LatLngPopup().add_to(map)
    inject_javascript(map)
    return map

def make_map(name="map_name", id="1"):
    map = Map(location=[39, 23], zoom_start=1)
    map._name, map._id = "map_name", "1"

    LatLngPopup().add_to(map)
    inject_javascript(map)
    fol = Folium(value=map, height=400, visible=False, elem_id='map-fol')
    return fol

def map_js():
    return  """
    (a, textBox) => {
        const iframeMap = document.getElementById('map-fol').getElementsByTagName('iframe')[0];
        const latlng = iframeMap.contentWindow.state_data;
        if (!latlng) { return; }
        textBox = `${latlng.lat},${latlng.lng}`;
        document.getElementById('coords-tbox').getElementsByTagName('textarea')[0].value = textBox;
        return [a, `${latlng.lat},${latlng.lng}`];
    }
    """

def haversine(lat1, lon1, lat2, lon2):
    if (lat1 is None) or (lon1 is None) or (lat2 is None) or (lon2 is None):
        return 0
    R = 6371  # radius of the earth in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = (
        sin(dLat / 2.0) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2.0) ** 2
    )
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance

def geoscore(d):
    return 5000 * exp(-d / 1492.7)


class Engine(object):
    def __init__(self, image_folder, csv_file, cache_path):
        self.image_folder = image_folder
        self.load_images_and_coordinates(csv_file)
        self.cache_path = cache_path
          
        # Initialize the score and distance lists
        self.index = 0
        self.stats = defaultdict(list)

        # Create the figure and canvas only once
        self.fig = plt.Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.MIN_LON, self.MAX_LON, self.MIN_LAT, self.MAX_LAT = self.ax.get_extent()

    def load_images_and_coordinates(self, csv_file):
        # Load the CSV
        df = pd.read_csv(csv_file)

        # Get the image filenames and their coordinates
        self.images = df['id'].tolist()[:]
        self.coordinates = df[['longitude', 'latitude']].values.tolist()[:]
        self.admins = df[['city', 'sub-region', 'region', 'country']].values.tolist()[:]
        self.preds = df[['pred_longitude', 'pred_latitude']].values.tolist()[:]

    def isfinal(self):
        return self.index == len(self.images)-1

    def load_image(self):
        if self.index > len(self.images)-1:          
            self.master.update_idletasks()
            self.finish()

        self.set_clock()
        return os.path.join(self.image_folder, f"{self.images[self.index]}.jpg"), '### ' + str(self.index + 1) + '/' + str(len(self.images))

    def get_figure(self):
        img_buf = io.BytesIO()
        self.fig.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        pil = Image.open(img_buf)
        self.width, self.height = pil.size
        return pil

    def normalize_pixels(self, click_lon, click_lat):
        return self.MIN_LON + click_lon * (self.MAX_LON-self.MIN_LON) / self.width, self.MIN_LAT + (self.height - click_lat+1) * (self.MAX_LAT-self.MIN_LAT) / self.height

    def set_clock(self):
        self.time = time.time()

    def get_clock(self):
        return time.time() - self.time

    def click(self, click_lon, click_lat):
        time_elapsed = self.get_clock()
        self.stats['times'].append(time_elapsed)

        # convert click_lon, click_lat to lat, lon (given that you have the borders of the image)
        # click_lon and click_lat is in pixels
        # lon and lat is in degrees
        # click_lon, click_lat = self.normalize_pixels(click_lon, click_lat)
        self.stats['clicked_locations'].append((click_lat, click_lon))
        true_lon, true_lat = self.coordinates[self.index]
        pred_lon, pred_lat = self.preds[self.index]

        self.ax.clear()
        self.ax.set_global()
        self.ax.stock_img()
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')

        self.ax.plot(pred_lon, pred_lat, 'gv', transform=ccrs.Geodetic(), label='model')
        self.ax.plot([true_lon, pred_lon], [true_lat, pred_lat], color='green', linewidth=1, transform=ccrs.Geodetic())
        self.ax.plot(click_lon, click_lat, 'bo', transform=ccrs.Geodetic(), label='user')
        self.ax.plot([true_lon, click_lon], [true_lat, click_lat], color='blue', linewidth=1, transform=ccrs.Geodetic())
        self.ax.plot(true_lon, true_lat, 'rx', transform=ccrs.Geodetic(), label='g.t.')
        legend = self.ax.legend(ncol=3, loc='lower center') #, bbox_to_anchor=(0.5, -0.15), borderaxespad=0.
        legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((1, 1, 1, 0.1))
        self.fig.canvas.draw()

        distance = haversine(true_lat, true_lon, click_lat, click_lon)
        score = geoscore(distance)
        self.stats['scores'].append(score)
        self.stats['distances'].append(distance)
        
        df = pd.DataFrame([self.get_model_average(who) for who in ['human', 'best', 'base']], columns=['who', 'GeoScore', 'Distance'])
        result_text = (f"### GeoScore: {score:.0f}, distance: {distance:.0f} km")

        self.cache(self.index+1, score, distance, (click_lat, click_lon), time_elapsed)
        return self.get_figure(), result_text, df

    def next_image(self):
        # Go to the next image
        self.index += 1
        return self.load_image()
        
    def get_model_average(self, which):
        if which == 'human':
            avg_score = sum(self.stats['scores']) / len(self.stats['scores']) if self.stats['scores'] else 0
            avg_distance = sum(self.stats['distances']) / len(self.stats['distances']) if self.stats['distances'] else 0
            return [which, avg_score, avg_distance]
        else:
            return [which, 0, 0]

    def update_average_display(self):
        # Calculate the average values
        avg_score = sum(self.stats['scores']) / len(self.stats['scores']) if self.stats['scores'] else 0
        avg_distance = sum(self.stats['distances']) / len(self.stats['distances']) if self.stats['distances'] else 0

        # Update the text box
        return f"GeoScore: {avg_score:.0f}, Distance: {avg_distance:.0f} km"
    
    def finish(self):
        clicks = rg.search(self.stats['clicked_locations'])
        clicked_admins = [[click['name'], click['admin2'], click['admin1'], click['cc']] for click in clicks]
        
        correct = [0,0,0,0]
        valid = [0,0,0,0]
        
        for clicked_admin, true_admin in zip(clicked_admins, self.admins):
            for i in range(4):
                if true_admin[i]!= 'nan':
                    valid[i] += 1
                if true_admin[i] == clicked_admin[i]:
                    correct[i] += 1
                    
        avg_city_accuracy = correct[0] / valid[0]
        avg_area_accuracy = correct[1] / valid[1]
        avg_region_accuracy = correct[2] / valid[2]
        avg_country_accuracy = correct[3] / valid[3]
        
        avg_score = sum(self.stats['scores']) / len(self.stats['scores']) if self.stats['scores'] else 0
        avg_distance = sum(self.stats['distances']) / len(self.stats['distances']) if self.stats['distances'] else 0

        final_results = (
            f"Average GeoScore: {avg_score:.0f}  \n" + 
            f"Average distance: {avg_distance:.0f} km  \n" + 
            f"Country Acc: {100*avg_country_accuracy:.1f}  \n" + 
            f"Region Acc: {100*avg_region_accuracy:.1f}  \n" + 
            f"Area Acc: {100*avg_area_accuracy:.1f}  \n" + 
            f"City Acc: {100*avg_city_accuracy:.1f}"
        )

        self.cache_final(final_results)

        # Update the text box
        return f"# Your stats 🌍\n" + final_results + f"  \n# Thanks for playing ❤️"
        
    # Function to save the game state
    def cache(self, index, score, distance, location, time_elapsed):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        with open(join(self.cache_path, str(index).zfill(2) + '.txt'), 'w') as f:
            print(f"{score}, {distance}, {location[0]}, {location[1]}, {time_elapsed}", file=f)

    # Function to save the game state
    def cache_final(self, final_results):
        times = ', '.join(map(str, self.stats['times']))
        fname = join(self.cache_path, 'full.txt')
        with open(fname, 'w') as f:
            print(f"{final_results}" + '\n Times: ' + times, file=f)

        zip_ = self.cache_path.rstrip('/') + '.zip'
        archived = shutil.make_archive(self.cache_path.rstrip('/'), 'zip', self.cache_path)
        try:
            wandb.init(project="plonk")
            artifact = wandb.Artifact('results', type='results')
            artifact.add_file(zip_)
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception:
            print("Failed to log results to wandb")
            pass

        if os.path.isfile(zip_):
            os.remove(zip_)

def make_page(engine):
    i = engine.index + 1
    total = len(engine.images)
    return f"<h3>{i}/{total}</h3>"


if __name__ == "__main__":
    # login with the key from secret
    wandb.login()
    if 'csv' in os.environ:
        csv_str = os.environ['csv']
        with open(CSV_FILE, 'w') as f:
            f.write(csv_str)

    import gradio as gr
    def click(state, coords):
        if coords == '-1' or state['clicked']:
            return gr.update(), gr.update(), gr.update(), gr.update()
        lat, lon = map(float, coords.split(','))
        state['clicked'] = True
        image, text, df = state['engine'].click(lon, lat)
        df = df.sort_values(by='GeoScore', ascending=False)
        return gr.update(visible=False), gr.update(value=image, visible=True), gr.update(value=text), gr.update(value=df, visible=True)

    def next_(state):
        if state['clicked']:
            if state['engine'].isfinal():
                text = state['engine'].finish()
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=text), gr.update(visible=False), gr.update(visible=False), gr.update(value="-1", visible=False)
            else:
                image, text = state['engine'].next_image()
                state['clicked'] = False
                return gr.update(value=make_map_(), visible=True), gr.update(visible=False), gr.update(value=image), gr.update(value=text), gr.update(), gr.update(), gr.update(visible=False), gr.update(value="-1")
        else:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def start(state):
        # create a unique random temporary name under CACHE_DIR
        # generate random hex and make sure it doesn't exist under CACHE_DIR
        while True:
            path = str(uuid.uuid4().hex)
            name = os.path.join(RESULTS_DIR, path)
            if not os.path.exists(name):
                break

        state['engine'] = Engine(IMAGE_FOLDER, CSV_FILE, name)
        state['clicked'] = False
        image, text = state['engine'].load_image()

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=image, visible=True),
            gr.update(value=text, visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value="<h1>OSV-5M (plonk)</h1>"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="-1"),
            gr.update(visible=True),
        )

 # head=space_js
    with gr.Blocks(css=css) as demo:
        state = gr.State({})
        rules = gr.Markdown(RULES, visible=True)


        start_button = gr.Button("Start", visible=True)
        with gr.Row():
            map_ = make_map()
            results = gr.Image(label='Results', visible=False)
            image_ = gr.Image(label='Image', visible=False)
        with gr.Row():
            text = gr.Markdown("", visible=False)
            text_count = gr.Markdown("", visible=False)

        with gr.Row():
            # map related
            select_button = gr.Button("Choose", elem_id='latlon_btn', visible=False)
            ####
            next_button = gr.Button("Next", visible=False, elem_id='next')
        perf = gr.Dataframe(value=None, visible=False)
    
        # map related
        coords = gr.Textbox(value="-1", label="Latitude, Longitude", visible=False, elem_id='coords-tbox')
        ####

        start_button.click(start, inputs=[state], outputs=[map_, results, image_, text_count, text, next_button, rules, state, start_button, coords, select_button])
        select_button.click(click, inputs=[state, coords], outputs=[map_, results, text, perf], js=map_js())
        next_button.click(next_, inputs=[state], outputs=[map_, results, image_, text_count, text, next_button, perf, coords])

    demo.launch(allowed_paths=["custom.ttf"], debug=True)
