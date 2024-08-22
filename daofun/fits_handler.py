from aplpy import FITSFigure
from astropy.visualization.wcsaxes import WCSAxes, WCSAxesSubplot

from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon, FancyArrow
from matplotlib.collections import PatchCollection, LineCollection
from astropy.wcs.utils import proj_plane_pixel_scales
import threading
import numpy as np

from functools import wraps

mydata = threading.local()

__all__ = ['auto_refresh', 'fixdocstring']


def auto_refresh(f):

    @wraps(f)
    def wrapper(*args, **kwargs):

        if 'refresh' in kwargs:
            refresh = kwargs.pop('refresh')
        else:
            refresh = True

        # The following is necessary rather than using mydata.nesting = 0 at the
        # start of the file, because doing the latter caused issues with the Django
        # development server.
        mydata.nesting = getattr(mydata, 'nesting', 0) + 1

        try:
            return f(*args, **kwargs)
        finally:
            mydata.nesting -= 1
            if hasattr(args[0], '_figure'):
                if refresh and mydata.nesting == 0 and args[0]._figure._auto_refresh:
                    args[0]._figure.canvas.draw()

    return wrapper

def uniformize_1d(*args):
    if len(args) > 1:
        return np.broadcast_arrays(np.atleast_1d(args[0]), *args[1:])
    elif len(args) == 1:
        return np.atleast_1d(args[0])
    else:
        raise ValueError("No arguments passed to uniformize_1d")

class FITSFigureV2(FITSFigure):
    def __init__(self, data, hdu=0, figure=None, subplot=(1, 1, 1),
                 downsample=False, north=False, convention=None,
                 dimensions=[0, 1], slices=[], auto_refresh=None,
                 **kwargs):
        super().__init__(data, hdu, figure, subplot, downsample, 
        					north, convention, dimensions, slices, 
                            auto_refresh, **kwargs)
    
    def new_subplot(self, subplot):
        self.ax = WCSAxesSubplot(self._figure, *subplot, wcs=self._wcs,
                                     slices=self._wcsaxes_slices)
        self._figure.add_subplot(self.ax)
		# Turn off autoscaling
        self.ax.set_autoscale_on(False)

        # Make sure axes are above everything else
        self.ax.set_axisbelow(False)
		
        # Set view to whole FITS file
        self._initialize_view()
		# Display minor ticks
        self.ax.coords[self.x].display_minor_ticks(True)
        self.ax.coords[self.y].display_minor_ticks(True)

        # Initialize layers list
        self._initialize_layers()

        # Set image holder to be empty
        self.image = None
    
    # Show circles. Different from markers as this method allows more
    # definitions for the circles.
    @auto_refresh
    def show_circles_colors(self, xw, yw, radius, layer=False, coords_frame='world', zorder=None, colors=None, **kwargs):
        """
        Overlay circles on the current plot.

        Parameters
        ----------

        xw : list or `~numpy.ndarray`
            The x positions of the centers of the circles (in world coordinates)

        yw : list or `~numpy.ndarray`
            The y positions of the centers of the circles (in world coordinates)

        radius : int or float or list or `~numpy.ndarray`
            The radii of the circles (in world coordinates)

        layer : str, optional
            The name of the circle layer. This is useful for giving
            custom names to layers (instead of circle_set_n) and for
            replacing existing layers.

        coords_frame : 'pixel' or 'world'
            The reference frame in which the coordinates are defined. This is
            used to interpret the values of ``xw`` and ``yw``.

        kwargs
            Additional keyword arguments (such as facecolor, edgecolor, alpha,
            or linewidth) are passed to Matplotlib
            :class:`~matplotlib.collections.PatchCollection` class, and can be
            used to control the appearance of the circles.
        """

        xw, yw, radius = uniformize_1d(xw, yw, radius)

        if 'facecolor' not in kwargs:
            kwargs.setdefault('facecolor', 'none')

        if layer:
            self.remove_layer(layer, raise_exception=False)

        if coords_frame not in ['pixel', 'world']:
            raise ValueError("coords_frame should be set to 'pixel' or 'world'")

        # While we could plot the shape using the get_transform('world') mode
        # from WCSAxes, the issue is that the rotation angle is also measured in
        # world coordinates so will not be what the user is expecting. So we allow the user to specify the reference frame for the coordinates and for the rotation.

        if coords_frame == 'pixel':
            x, y = xw, yw
            r = radius
        else:
            x, y = self.world2pixel(xw, yw)
            pix_scale = proj_plane_pixel_scales(self._wcs)
            sx, sy = pix_scale[self.x], pix_scale[self.y]
            r = radius / np.sqrt(sx * sy)

        patches = []
        for i in range(len(xw)):
            patches.append(Circle((x[i], y[i]), radius=r[i]))

        # Due to bugs in matplotlib, we need to pass the patch properties
        # directly to the PatchCollection rather than use match_original.
        p = PatchCollection(patches, **kwargs)
        if not isinstance(colors, type(None)):
            p.set_color(colors)

        if zorder is not None:
            p.zorder = zorder
        c = self.ax.add_collection(p)

        if layer:
            circle_set_name = layer
        else:
            self._circle_counter += 1
            circle_set_name = 'circle_set_' + str(self._circle_counter)

        self._layers[circle_set_name] = c
