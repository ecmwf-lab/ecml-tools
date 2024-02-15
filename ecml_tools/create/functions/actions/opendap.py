# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import numpy as np
from climetlab import load_source
from climetlab.utils.patterns import Pattern
from scipy.spatial import  KDTree


def latlon_to_xyz(lat, lon, radius=1.0):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    # We assume that the Earth is a sphere of radius 1 so N(phi) = 1
    # We assume h = 0
    #
    phi = np.deg2rad(lat)
    lda = np.deg2rad(lon)

    cos_phi = np.cos(phi)
    cos_lda = np.cos(lda)
    sin_phi = np.sin(phi)
    sin_lda = np.sin(lda)

    x = cos_phi * cos_lda * radius
    y = cos_phi * sin_lda * radius
    z = sin_phi * radius

    return x, y, z


class Triangle3D:

    def __init__(self, v0, v1, v2):

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def intersect(self, ray_origin, ray_direction):
        # Möller–Trumbore intersection algorithm
        # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

        epsilon = 0.0000001

        h = np.cross(ray_direction, self.v2 - self.v0)
        a = np.dot(self.v1 - self.v0, h)

        if -epsilon < a < epsilon:
            return None

        f = 1.0 / a
        s = ray_origin - self.v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, self.v1 - self.v0)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(self.v2 - self.v0, q)

        if t > epsilon:
            return t

        return None


def opendap(context, dates, url_pattern, *args, **kwargs):

    all_urls = Pattern(url_pattern).substitute(*args, date=dates, **kwargs)

    for url in all_urls:

        print("URL", url)
        ds = load_source("opendap", url)
        x = ds.to_xarray()
        lats = x.latitude.data.flatten()
        lons = x.longitude.data.flatten()

        north = np.amax(lats)
        south = np.amin(lats)
        east = np.amax(lons)
        west = np.amin(lons)

        era_ds = load_source(
            "mars",
            {
                "class": "ea",
                "date": -200,
                "levtype": "sfc",
                "param": "2t",
                "area": [
                    np.min([90.0, north + 2]),
                    west - 2,
                    np.max([-90.0, south - 2]),
                    east + 2,
                ],
            },
        )
        era_lats, era_lons = era_ds[0].grid_points()
        era_xyx = latlon_to_xyz(era_lats, era_lons)
        era_points = np.array(era_xyx).transpose()

        xyx = latlon_to_xyz(lats, lons)
        points = np.array(xyx).transpose()

        print("make tree")
        kdtree = KDTree(points)
        print("query")
        distances, indices = kdtree.query(era_points, k=3)
        print("done")

        zero = np.array([0.0, 0.0, 0.0])
        ok = []
        for i, (era_point, distance, index) in enumerate(
            zip(era_points, distances, indices)
        ):
            t = Triangle3D(points[index[0]], points[index[1]], points[index[2]])
            if not t.intersect(zero, era_point):
                ok.append(i)

        ok = np.array(ok)
        print(ok.shape)

        import matplotlib.pyplot as plt

        plt.scatter(era_lons[ok] - 360, era_lats[ok], s=0.01, c="r")
        # plt.scatter(lons, lats, s=0.01)

        plt.savefig("era1.png")
        plt.scatter(lons, lats, s=0.01)
        plt.savefig("era2.png")


execute = opendap
