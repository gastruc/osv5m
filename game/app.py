"""Requires gradio==4.27.0"""
import io
import shutil 
import os
import json
import uuid
import time
import math
import datetime
import numpy as np

from uuid import uuid4
from PIL import Image
from math import radians, sin, cos, sqrt, asin, exp
from os.path import join
from collections import defaultdict
from itertools import tee

import matplotlib.style as mplstyle
mplstyle.use(['fast'])
import pandas as pd

import gradio as gr
import reverse_geocoder as rg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from gradio_folium import Folium
from geographiclib.geodesic import Geodesic
from folium import Map, Element, LatLngPopup, Marker, PolyLine, FeatureGroup
from folium.map import LayerControl
from folium.plugins import BeautifyIcon
from huggingface_hub import CommitScheduler

MPL = False
IMAGE_FOLDER = './images'
CSV_FILE = './select.csv'
BASE_LOCATION = [0, 23]
RULES = """<h1 style="margin-bottom: 0.5em">OSV-5M (plonk)</h1>
<center style="margin-bottom: 1em; margin-top: 1em"><img width="256" alt="Rotating globe" src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Rotating_globe.gif"></center>
<h2 style="margin-top: 0.5em"> Instructions </h2>
<h3> Click on the map 🗺️ (left) to the location at which you think the image 🖼️ (right) was captured!</h3>
<h3 style="margin-bottom: 0.5em"> Click "Select" to finalize your selection and then "Next" to move to the next image.</h3>

<h2> AI Competitors </h2>
<h3> You will compete against two AIs: <b>Plonk-AI</b> (our best model) and Baseline-AI (a simpler approach).</h3>
<h3> These AIs have not been trained on any of the images you will see; in fact, they haven't seen anything within a <b>1km radius</b> of them.</h3>
<h3 style="margin-bottom: 0.5em"> Like you, the AIs will need to pick up on geographic clues to pinpoint the locations of the images.</h3>

<h2> Geoscore </h2>
<h3> The geoscore is calculated based on how close each guess is to the true location as in Geoguessr, with a maximum of <b>5000 points: $$\\large g(d) = 5000 \\exp\\left(\\frac{-d}{1492.7}\\right)$$ </h3>
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
    font-size: 3.2em;
}
img {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
    font-family: custom;
    font-size: 2.2em;
}
h3 {
    text-align: center;
    display:block;
    font-family: custom;
    font-weight: normal;
    font-size: 1.5em;
}

.MathJax {
    font-size: 1.5em;
}
"""

space_js = """
<script src="https://cdn.jsdelivr.net/npm/@rapideditor/country-coder@5.2/dist/country-coder.iife.min.js"></script>
<script type="text/javascript"
  src="http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
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

function shortcuts_exit(e) {
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        break;
        default:
        if (e.key.toLowerCase() == "e" && e.shiftKey) {
            document.getElementById("exit_btn").click();
        }
    }
}
document.addEventListener('keypress', shortcuts, false);
document.addEventListener('keypress', shortcuts_exit, false);
</script>
"""

def sample_points_along_geodesic(start_lat, start_lon, end_lat, end_lon, min_length_km=2000, segment_length_km=5000, num_samples=None):
    geod = Geodesic.WGS84
    distance = geod.Inverse(start_lat, start_lon, end_lat, end_lon)['s12']
    if distance < min_length_km:
        return [(start_lat, start_lon), (end_lat, end_lon)]

    if num_samples is None:
        num_samples = min(int(distance / segment_length_km) + 1, 1000)
    point_distance = np.linspace(0, distance, num_samples)
    points = []
    for pd in point_distance:
        line = geod.InverseLine(start_lat, start_lon, end_lat, end_lon)
        g_point = line.Position(pd, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        points.append((g_point['lat2'], g_point['lon2']))
    return points

class GeodesicPolyLine(PolyLine):
    def __init__(self, locations, min_length_km=2000, segment_length_km=1000, num_samples=None, **kwargs):
        kwargs1 = dict(min_length_km=min_length_km, segment_length_km=segment_length_km, num_samples=num_samples)
        assert len(locations) == 2, "A polyline must have at least two locations"
        start, end = locations
        geodesic_locs = sample_points_along_geodesic(start[0], start[1], end[0], end[1], **kwargs1)
        super().__init__(geodesic_locs, **kwargs)

def inject_javascript(folium_map):
    js = """
    document.addEventListener('DOMContentLoaded', function() {
        map_name_1.on('click', function(e) {
            window.state_data = e.latlng
        });
    });
    """
    folium_map.get_root().html.add_child(Element(f'<script>{js}</script>'))

def empty_map():
    return Map(location=BASE_LOCATION, zoom_start=1)

def make_map_(name="map_name", id="1"):
    map = Map(location=BASE_LOCATION, zoom_start=1)
    map._name, map._id = name, id

    LatLngPopup().add_to(map)
    inject_javascript(map)
    return map

def make_map(name="map_name", id="1", height=500):
    map = make_map_(name, id)
    fol = Folium(value=map, height=height, visible=False, elem_id='map-fol')
    return fol

def map_js():
    return  """
    (a, textBox) => {
        const iframeMap = document.getElementById('map-fol').getElementsByTagName('iframe')[0];
        const latlng = iframeMap.contentWindow.state_data;
        if (!latlng) { return [-1, -1]; }
        textBox = `${latlng.lat},${latlng.lng}`;
        document.getElementById('coords-tbox').getElementsByTagName('textarea')[0].value = textBox;
        var a = countryCoder.iso1A2Code([latlng.lng, latlng.lat]);
        if (!a) { a = 'nan'; }
        return [a, `${latlng.lat},${latlng.lng},${a}`];
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

def compute_scores(csv_file):
    df = pd.read_csv(csv_file)
    if 'accuracy_country' not in df.columns:
        print('Computing scores... (this may take a while)')
        geocoders = rg.search([(row.true_lat, row.true_lon) for row in df.itertuples(name='Pandas')])
        df['city'] = [geocoder['name'] for geocoder in geocoders]
        df['area'] = [geocoder['admin2'] for geocoder in geocoders]
        df['region'] = [geocoder['admin1'] for geocoder in geocoders]
        df['country'] = [geocoder['cc'] for geocoder in geocoders]

        df['city_val'] = df['city'].apply(lambda x: 0 if pd.isna(x) or x == 'nan' else 1)
        df['area_val'] = df['area'].apply(lambda x: 0 if pd.isna(x) or x == 'nan' else 1)
        df['region_val'] = df['region'].apply(lambda x: 0 if pd.isna(x) or x == 'nan' else 1)
        df['country_val'] = df['country'].apply(lambda x: 0 if pd.isna(x) or x == 'nan' else 1)

        df['distance'] = df.apply(lambda row: haversine(row['true_lat'], row['true_lon'], row['pred_lat'], row['pred_lon']), axis=1)
        df['score'] = df.apply(lambda row: geoscore(row['distance']), axis=1)
        df['distance_base'] = df.apply(lambda row: haversine(row['true_lat'], row['true_lon'], row['pred_lat_base'], row['pred_lon_base']), axis=1)
        df['score_base'] = df.apply(lambda row: geoscore(row['distance_base']), axis=1)

        print('Computing geocoding accuracy (base)...')
        geocoders_base = rg.search([(row.pred_lat_base, row.pred_lon_base) for row in df.itertuples(name='Pandas')])
        df['pred_city_base'] = [geocoder['name'] for geocoder in geocoders_base]
        df['pred_area_base'] = [geocoder['admin2'] for geocoder in geocoders_base]
        df['pred_region_base'] = [geocoder['admin1'] for geocoder in geocoders_base]
        df['pred_country_base'] = [geocoder['cc'] for geocoder in geocoders_base]
    
        df['city_hit_base'] = [df['city'].iloc[i] != 'nan' and df['pred_city_base'].iloc[i] == df['city'].iloc[i] for i in range(len(df))]
        df['area_hit_base'] = [df['area'].iloc[i] != 'nan' and df['pred_area_base'].iloc[i] == df['area'].iloc[i] for i in range(len(df))]
        df['region_hit_base'] = [df['region'].iloc[i] != 'nan' and df['pred_region_base'].iloc[i] == df['region'].iloc[i] for i in range(len(df))]
        df['country_hit_base'] = [df['country'].iloc[i] != 'nan' and df['pred_country_base'].iloc[i] == df['country'].iloc[i] for i in range(len(df))]

        df['accuracy_city_base'] = [(0 if df['city_val'].iloc[:i].sum() == 0 else df['city_hit_base'].iloc[:i].sum()/df['city_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_area_base'] = [(0 if df['area_val'].iloc[:i].sum() == 0 else df['area_hit_base'].iloc[:i].sum()/df['area_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_region_base'] = [(0 if df['region_val'].iloc[:i].sum() == 0 else df['region_hit_base'].iloc[:i].sum()/df['region_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_country_base'] = [(0 if df['country_val'].iloc[:i].sum() == 0 else df['country_hit_base'].iloc[:i].sum()/df['country_val'].iloc[:i].sum())*100 for i in range(len(df))]

        print('Computing geocoding accuracy (best)...')
        geocoders = rg.search([(row.pred_lat, row.pred_lon) for row in df.itertuples()])
        df['pred_city'] = [geocoder['name'] for geocoder in geocoders]
        df['pred_area'] = [geocoder['admin2'] for geocoder in geocoders]
        df['pred_region'] = [geocoder['admin1'] for geocoder in geocoders]
        df['pred_country'] = [geocoder['cc'] for geocoder in geocoders]
    
        df['city_hit'] = [df['city'].iloc[i] != 'nan' and df['pred_city'].iloc[i] == df['city'].iloc[i] for i in range(len(df))]
        df['area_hit'] = [df['area'].iloc[i] != 'nan' and df['pred_area'].iloc[i] == df['area'].iloc[i] for i in range(len(df))]
        df['region_hit'] = [df['region'].iloc[i] != 'nan' and df['pred_region'].iloc[i] == df['region'].iloc[i] for i in range(len(df))]
        df['country_hit'] = [df['country'].iloc[i] != 'nan' and df['pred_country'].iloc[i] == df['country'].iloc[i] for i in range(len(df))]

        df['accuracy_city'] = [(0 if df['city_val'].iloc[:i].sum() == 0 else df['city_hit'].iloc[:i].sum()/df['city_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_area'] = [(0 if df['area_val'].iloc[:i].sum() == 0 else df['area_hit'].iloc[:i].sum()/df['area_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_region'] = [(0 if df['region_val'].iloc[:i].sum() == 0 else df['region_hit'].iloc[:i].sum()/df['region_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df['accuracy_country'] = [(0 if df['country_val'].iloc[:i].sum() == 0 else df['country_hit'].iloc[:i].sum()/df['country_val'].iloc[:i].sum())*100 for i in range(len(df))]
        df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    JSON_DATASET_DIR = 'results'
    scheduler = CommitScheduler(
        repo_id="osv5m/humeval",
        repo_type="dataset",
        folder_path=JSON_DATASET_DIR,
        path_in_repo=f"raw_data",
        every=2
    )


class Engine(object):
    def __init__(self, image_folder, csv_file, mpl=True):
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.load_images_and_coordinates(csv_file)
          
        # Initialize the score and distance lists
        self.index = 0
        self.stats = defaultdict(list)

        # Create the figure and canvas only once
        self.fig = plt.Figure(figsize=(10, 6))
        self.mpl = mpl
        if mpl:
            self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        self.tag = str(uuid4()) + datetime.datetime.now().strftime("__%Y_%m_%d_%H_%M_%S")

    def load_images_and_coordinates(self, csv_file):
        # Load the CSV
        df = pd.read_csv(csv_file)
        # Put image with id 732681614433401 on the top and then all the rest below
        df['id'] = df['id'].astype(str)
        df = pd.concat([df[df['id'] == '495204901603170'], df[df['id'] != '495204901603170']])
        df = pd.concat([df[df['id'] == '732681614433401'], df[df['id'] != '732681614433401']])

        # Get the image filenames and their coordinates
        self.images = [os.path.join(self.image_folder, f"{img_path}.jpg") for img_path in df['id'].tolist()[:]]
        self.coordinates = df[['true_lon', 'true_lat']].values.tolist()[:]

        # compute the admins
        self.df = df
        self.admins = self.df[['city', 'area', 'region', 'country']].values.tolist()[:]
        self.preds = self.df[['pred_lon', 'pred_lat']].values.tolist()[:]

    def isfinal(self):
        return self.index == len(self.images)-1

    def load_image(self):
        if self.index > len(self.images)-1:          
            self.master.update_idletasks()
            self.finish()

        self.set_clock()
        return self.images[self.index], '### ' + str(self.index + 1) + '/' + str(len(self.images))

    def get_figure(self):
        if self.mpl:
            img_buf = io.BytesIO()
            self.fig.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
            pil = Image.open(img_buf)
            self.width, self.height = pil.size
            return pil
        else:
            pred_lon, pred_lat, true_lon, true_lat, click_lon, click_lat = self.info
            map = Map(location=BASE_LOCATION, zoom_start=1)
            map._name, map._id = 'visu', '1'

            feature_group = FeatureGroup(name='Ground Truth')
            Marker(
                location=[true_lat, true_lon],
                popup="True location",
                icon_color='red',
            ).add_to(feature_group)
            map.add_child(feature_group)

            icon_square = BeautifyIcon(
                icon_shape='rectangle-dot', 
                border_color='green', 
                border_width=5,
            )
            feature_group_best = FeatureGroup(name='Best Model')
            Marker(
                location=[pred_lat, pred_lon],
                popup="Best Model",
                icon=icon_square,
            ).add_to(feature_group_best)
            GeodesicPolyLine([[true_lat, true_lon], [pred_lat, pred_lon]], color='green').add_to(feature_group_best)
            map.add_child(feature_group_best)

            icon_circle = BeautifyIcon(
                icon_shape='circle-dot', 
                border_color='blue', 
                border_width=5,
            )
            feature_group_user = FeatureGroup(name='User')
            Marker(
                location=[click_lat, click_lon],
                popup="Human",
                icon=icon_circle,
            ).add_to(feature_group_user)
            GeodesicPolyLine([[true_lat, true_lon], [click_lat, click_lon]], color='blue').add_to(feature_group_user)
            map.add_child(feature_group_user)

            map.add_child(LayerControl())

            return map

    def set_clock(self):
        self.time = time.time()

    def get_clock(self):
        return time.time() - self.time

    def mpl_style(self, pred_lon, pred_lat, true_lon, true_lat, click_lon, click_lat):
        if self.mpl:
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
            self.fig.canvas.draw()
        else:
            self.info = [pred_lon, pred_lat, true_lon, true_lat, click_lon, click_lat]


    def click(self, click_lon, click_lat, country):
        time_elapsed = self.get_clock()
        self.stats['times'].append(time_elapsed)

        # convert click_lon, click_lat to lat, lon (given that you have the borders of the image)
        # click_lon and click_lat is in pixels
        # lon and lat is in degrees
        self.stats['clicked_locations'].append((click_lat, click_lon))
        true_lon, true_lat = self.coordinates[self.index]
        pred_lon, pred_lat = self.preds[self.index]
        self.mpl_style(pred_lon, pred_lat, true_lon, true_lat, click_lon, click_lat)

        distance = haversine(true_lat, true_lon, click_lat, click_lon)
        score = geoscore(distance)
        self.stats['scores'].append(score)
        self.stats['distances'].append(distance)
        self.stats['country'].append(int(self.admins[self.index][3] != 'nan' and country == self.admins[self.index][3]))

        df = pd.DataFrame([self.get_model_average(who) for who in ['user', 'best', 'base']], columns=['who', 'GeoScore', 'Distance', 'Accuracy (country)']).round(2)
        result_text = (
            f"### <span style='color:blue'>GeoScore: %s, Distance: %s km <b style='color:blue'>(You)</b></span></br><span style='color:green'>GeoScore: %s, Distance: %s km <b style='color:green'>(Plonk-AI)</b></span>" % (
                round(score, 2),
                round(distance, 2),
                round(self.df['score'].iloc[self.index], 2),
                round(self.df['distance'].iloc[self.index], 2)
            )
        )
        # You: }   \green{OSV-Bot:  GeoScore: XX, distance: XX

        self.cache(self.index+1, score, distance, (click_lat, click_lon), time_elapsed)
        return self.get_figure(), result_text, df

    def next_image(self):
        # Go to the next image
        self.index += 1
        return self.load_image()

    def get_model_average(self, which, all=False, final=False):
        aux, i = [], self.index
        if which == 'user':
            avg_score = sum(self.stats['scores']) / len(self.stats['scores']) if self.stats['scores'] else 0
            avg_distance = sum(self.stats['distances']) / len(self.stats['distances']) if self.stats['distances'] else 0
            avg_country_accuracy = (0 if self.df['country_val'].iloc[:i+1].sum() == 0 else sum(self.stats['country'])/self.df['country_val'].iloc[:i+1].sum())*100
            if all:
                avg_city_accuracy = (0 if self.df['city_val'].iloc[:i+1].sum() == 0 else sum(self.stats['city'])/self.df['city_val'].iloc[:i+1].sum())*100
                avg_area_accuracy = (0 if self.df['area_val'].iloc[:i+1].sum() == 0 else sum(self.stats['area'])/self.df['area_val'].iloc[:i+1].sum())*100
                avg_region_accuracy = (0 if self.df['region_val'].iloc[:i+1].sum() == 0 else sum(self.stats['region'])/self.df['region_val'].iloc[:i+1].sum())*100
                aux = [avg_city_accuracy, avg_area_accuracy, avg_region_accuracy]
            which = 'You'
        elif which == 'base':
            avg_score = np.mean(self.df[['score_base']].iloc[:i+1])
            avg_distance = np.mean(self.df[['distance_base']].iloc[:i+1])
            avg_country_accuracy = self.df['accuracy_country_base'].iloc[i]
            if all:
                aux = [self.df['accuracy_city_base'].iloc[i], self.df['accuracy_area_base'].iloc[i], self.df['accuracy_region_base'].iloc[i]]
            which = 'Baseline-AI'
        elif which == 'best':
            avg_score = np.mean(self.df[['score']].iloc[:i+1])
            avg_distance = np.mean(self.df[['distance']].iloc[:i+1])
            avg_country_accuracy = self.df['accuracy_country'].iloc[i]
            if all:
                aux = [self.df['accuracy_city_base'].iloc[i], self.df['accuracy_area_base'].iloc[i], self.df['accuracy_region_base'].iloc[i]]
            which = 'Plonk-AI'
        return [which, avg_score, avg_distance, avg_country_accuracy] + aux

    def update_average_display(self):
        # Calculate the average values
        avg_score = sum(self.stats['scores']) / len(self.stats['scores']) if self.stats['scores'] else 0
        avg_distance = sum(self.stats['distances']) / len(self.stats['distances']) if self.stats['distances'] else 0

        # Update the text box
        return f"GeoScore: {avg_score:.0f}, Distance: {avg_distance:.0f} km"
    
    def finish(self):
        clicks = rg.search(self.stats['clicked_locations'])
        self.stats['city'] = [(int(self.admins[self.index][0] != 'nan' and click['name'] == self.admins[self.index][0])) for click in clicks]
        self.stats['area'] = [(int(self.admins[self.index][1] != 'nan' and click['admin2'] == self.admins[self.index][1])) for click in clicks]
        self.stats['region'] = [(int(self.admins[self.index][2] != 'nan' and click['admin1'] == self.admins[self.index][2])) for click in clicks]
        
        df = pd.DataFrame([self.get_model_average(who, True, True) for who in ['user', 'best', 'base']], columns=['who', 'GeoScore', 'Distance', 'Accuracy (country)', 'Accuracy (city)', 'Accuracy (area)', 'Accuracy (region)'])
        return df
        
    # Function to save the game state
    def cache(self, index, score, distance, location, time_elapsed):
        with scheduler.lock:
            os.makedirs(join(JSON_DATASET_DIR, self.tag), exist_ok=True)
            with open(join(JSON_DATASET_DIR, self.tag, f'{index}.json'), 'w') as f:
                json.dump({"lat": location[0], "lon": location[1], "time": time_elapsed, "user": self.tag}, f)
                f.write('\n')


if __name__ == "__main__":
    # login with the key from secret
    if 'csv' in os.environ:
        csv_str = os.environ['csv']
        with open(CSV_FILE, 'w') as f:
            f.write(csv_str)
    
    compute_scores(CSV_FILE)
    import gradio as gr
    def click(state, coords):
        if coords == '-1' or state['clicked']:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        lat, lon, country = coords.split(',')
        state['clicked'] = True
        image, text, df = state['engine'].click(float(lon), float(lat), country)
        df = df.sort_values(by='GeoScore', ascending=False)
        kargs = {}
        if not MPL:
            kargs = {'value': empty_map()}
        return gr.update(visible=False, **kargs), gr.update(value=image, visible=True), gr.update(value=text, visible=True), gr.update(value=df, visible=True), gr.update(visible=False), gr.update(visible=True),

    def exit_(state):
        if state['engine'].index > 0:
            df = state['engine'].finish()
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value='', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=df, visible=True), gr.update(value="-1", visible=False), gr.update(value="<h1 style='margin-top: 4em;'> Your stats on OSV-5M🌍 </h1>", visible=True), gr.update(value="<h3 style='margin-top: 1em;'>Thanks for playing ❤️</h3>", visible=True), gr.update(visible=False)
        else:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def next_(state):
        if state['clicked']:
            if state['engine'].isfinal():
                return exit_(state)
            else:
                image, text = state['engine'].next_image()
                state['clicked'] = False
                kargs = {}
                if not MPL:
                    kargs = {'value': empty_map()}
                return gr.update(value=make_map_(), visible=True), gr.update(visible=False, **kargs), gr.update(value=image), gr.update(value=text, visible=True), gr.update(value='', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="-1"), gr.update(), gr.update(), gr.update(visible=True)
        else:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def start(state):
        # create a unique random temporary name under CACHE_DIR
        # generate random hex and make sure it doesn't exist under CACHE_DIR
        state['engine'] = Engine(IMAGE_FOLDER, CSV_FILE, MPL)
        state['clicked'] = False
        image, text = state['engine'].load_image()

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=image, visible=True),
            gr.update(value=text, visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="<h1>OSV-5M (plonk)</h1>"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="-1"),
            gr.update(visible=True),
        )

    with gr.Blocks(css=css, head=space_js) as demo:
        state = gr.State({})
        rules = gr.Markdown(RULES, visible=True)

        exit_button = gr.Button("Exit", visible=False, elem_id='exit_btn')
        start_button = gr.Button("Start", visible=True)
        with gr.Row():
            map_ = make_map(height=512)
            if MPL:
                results = gr.Image(label='Results', visible=False)
            else:
                results = Folium(height=512, visible=False)
            image_ = gr.Image(label='Image', visible=False, height=512)

        with gr.Row():
            text = gr.Markdown("", visible=False)
            text_count = gr.Markdown("", visible=False)

        with gr.Row():
            select_button = gr.Button("Select", elem_id='latlon_btn', visible=False)
            next_button = gr.Button("Next", visible=False, elem_id='next')
        perf = gr.Dataframe(value=None, visible=False, label='Average Performance')
        text_end = gr.Markdown("", visible=False)
    
        coords = gr.Textbox(value="-1", label="Latitude, Longitude", visible=False, elem_id='coords-tbox')
        start_button.click(start, inputs=[state], outputs=[map_, results, image_, text_count, text, next_button, rules, state, start_button, coords, select_button])
        select_button.click(click, inputs=[state, coords], outputs=[map_, results, text, perf, select_button, next_button], js=map_js())
        next_button.click(next_, inputs=[state], outputs=[map_, results, image_, text_count, text, next_button, perf, coords, rules, text_end, select_button])
        exit_button.click(exit_, inputs=[state], outputs=[map_, results, image_, text_count, text, next_button, perf, coords, rules, text_end, select_button])

    demo.queue().launch(allowed_paths=["custom.ttf"], debug=True)
