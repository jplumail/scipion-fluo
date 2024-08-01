#  **************************************************************************
# *
# * Authors:     Jean Plumail (jplumail@unistra.fr) [1]
# *
# * [1] ICube, Université de Strasbourg
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from __future__ import annotations

import json
import math
import os
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    cast,
)

import numpy as np
import pint
import pyworkflow.object as pwobj

# type: ignore
import pyworkflow.utils as pwutils  # type: ignore
import tifffile  # type: ignore
from numpy.typing import NDArray
from ome_types import from_xml
from ome_types.model.simple_types import UnitsLength
from pyworkflow.object import (
    CsvList,
    Integer,
    Object,
    Pointer,
    Scalar,
    Set,
    String,
)
from scipy.ndimage import zoom

from pwfluo.constants import MICRON_STR


class FluoObject(pwobj.Object):
    """Simple base class from which all objects in this Domain will
    inherit from.
    """

    pass


class Matrix(Scalar, FluoObject):
    def __init__(self, **kwargs) -> None:
        Scalar.__init__(self, **kwargs)
        self._matrix: NDArray[np.float64] = np.eye(4)

    def _convertValue(self, value: str) -> None:
        """Value should be a str with comma separated values
        or a list.
        """
        self._matrix = np.array(json.loads(value)).astype(np.float64)

    def getObjValue(self) -> str:
        self._objValue = json.dumps(self._matrix.tolist())
        return self._objValue

    def setValue(self, i: int, j: int, value: float) -> None:
        self._matrix[i, j] = value

    def getMatrix(self) -> NDArray[np.float64]:
        """Return internal numpy matrix."""
        return self._matrix

    def setMatrix(self, matrix: NDArray[np.float64]) -> None:
        """Override internal numpy matrix."""
        self._matrix = matrix

    def __str__(self) -> str:
        return np.array_str(self._matrix)

    def _copy(self, other: Matrix, *args, **kwargs) -> None:
        """Override the default behaviour of copy
        to also copy array data.
        Copy other into self.
        """
        self.setMatrix(np.copy(other.getMatrix()))
        self._objValue = other._objValue


class Transform(FluoObject):
    """This class will contain a transformation matrix
    that can be applied to 2D/3D objects like images and volumes.
    It should contain information about euler angles, translation(or shift)
    and mirroring.
    Shifts are stored in pixels as treated in extract coordinates, or assign angles,...
    """

    # Basic Transformation factory
    ROT_X_90_CLOCKWISE = "rotX90c"
    ROT_Y_90_CLOCKWISE = "rotY90c"
    ROT_Z_90_CLOCKWISE = "rotZ90c"
    ROT_X_90_COUNTERCLOCKWISE = "rotX90cc"
    ROT_Y_90_COUNTERCLOCKWISE = "rotY90cc"
    ROT_Z_90_COUNTERCLOCKWISE = "rotZ90cc"

    def __init__(self, matrix: NDArray[np.float64] | None = None, **kwargs):
        FluoObject.__init__(self, **kwargs)
        self._matrix = Matrix()
        if matrix is not None:
            self.setMatrix(matrix)

    def getMatrix(self) -> NDArray[np.float64]:
        return self._matrix.getMatrix()

    def getRotationMatrix(self) -> NDArray[np.float64]:
        M = self.getMatrix()
        return M[:3, :3]

    def getShifts(self) -> NDArray[np.float64]:
        M = self.getMatrix()
        return M[:3, 3]

    def getMatrixAsList(self) -> list:
        """Return the values of the Matrix as a list."""
        return self._matrix.getMatrix().flatten().tolist()

    def setMatrix(self, matrix: NDArray[np.float64]):
        self._matrix.setMatrix(matrix)

    def __str__(self) -> str:
        return str(self._matrix)

    def scale(self, factor: float) -> None:
        m = self.getMatrix()
        m *= factor
        m[3, 3] = 1.0

    def scaleShifts(self, factor: float) -> None:
        # By default Scipion uses a coordinate system associated
        # with the volume rather than the projection
        m = self.getMatrix()
        m[:3, 3] *= factor

    def invert(self) -> Matrix:
        """Inverts the transformation and returns the matrix"""
        self._matrix.setMatrix(np.linalg.inv(self._matrix.getMatrix()))
        return self._matrix

    def setShifts(self, x: float, y: float, z: float) -> None:
        m = self.getMatrix()
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z

    def setShiftsTuple(self, shifts: tuple[float, float, float]) -> None:
        self.setShifts(shifts[0], shifts[1], shifts[2])

    def composeTransform(self, matrix: np.ndarray) -> None:
        """Apply a transformation matrix to the current matrix"""
        new_matrix = np.matmul(matrix, self.getMatrix())
        # new_matrix = matrix * self.getMatrix()
        self._matrix.setMatrix(new_matrix)

    @classmethod
    def create(cls, type: str) -> Transform:
        """Creates a default Transform object.
        Type is a string: `rot[X,Y,Z]90[c,cc]`
        with `c` meaning clockwise and `cc` counter clockwise
        """
        if type == cls.ROT_X_90_CLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
                )
            )
        elif type == cls.ROT_X_90_COUNTERCLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                )
            )
        elif type == cls.ROT_Y_90_CLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[1, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
                )
            )
        elif type == cls.ROT_Y_90_COUNTERCLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
                )
            )
        elif type == cls.ROT_Z_90_CLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
            )
        elif type == cls.ROT_Z_90_COUNTERCLOCKWISE:
            return Transform(
                matrix=np.array(
                    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
            )
        else:
            TRANSFORMATION_FACTORY_TYPES = [
                cls.ROT_X_90_CLOCKWISE,
                cls.ROT_Y_90_CLOCKWISE,
                cls.ROT_Z_90_CLOCKWISE,
                cls.ROT_X_90_COUNTERCLOCKWISE,
                cls.ROT_Y_90_COUNTERCLOCKWISE,
                cls.ROT_Z_90_COUNTERCLOCKWISE,
            ]
            raise Exception(
                "Introduced Transformation type is not recognized.\n"
                "Admitted values are\n"
                "%s" % " ".join(TRANSFORMATION_FACTORY_TYPES)
            )


class ImageDim(CsvList, FluoObject):
    """Just a wrapper to a CsvList to store image dimensions
    as X, Y and Z.
    """

    def __init__(
        self, x: int | None = None, y: int | None = None, z: int | None = None
    ) -> None:
        CsvList.__init__(self, pType=int)
        if x is not None and y is not None:
            self.append(x)
            self.append(y)
            if z is not None:
                self.append(z)

    def getX(self) -> int | None:
        if self.isEmpty():
            return None
        return self[0]

    def getY(self) -> int | None:
        if self.isEmpty():
            return None
        return self[1]

    def getZ(self) -> int | None:
        if self.isEmpty():
            return None
        return self[2]

    def set_(self, dims: tuple[int, int, int] | None) -> None:
        if dims is not None:
            if all(isinstance(dims[i], int) for i in range(3)):
                if self.isEmpty():
                    for i in range(3):
                        self.append(dims[i])
                else:
                    self[:] = dims
            else:
                raise Exception(
                    "Dimensions must be a tuple of int, "
                    f"got {dims} of type {type(dims)}"
                )
        else:
            self.clear()

    def __str__(self) -> str:
        x, y, z = self.getX(), self.getY(), self.getZ()
        if (x is None) or (y is None) or (z is None):
            s = "No-Dim"
        else:
            s = "%d x %d x %d" % (x, y, z)
        return s


class VoxelSize(CsvList, FluoObject):
    """Just a wrapper to a CsvList to store a voxel size
    as XY and Z.
    """

    def __init__(self, xy: float | None = None, z: float | None = None):
        CsvList.__init__(self, pType=float)
        if xy is not None and z is not None:
            self.append(xy)
            self.append(z)

    def setVoxelSize(self, xy: float, z: float) -> None:
        if self.isEmpty():
            self.append(xy)
            self.append(z)
        else:
            self[0], self[1] = xy, z

    def getVoxelSize(self) -> tuple[float, float] | None:
        """returns voxel size in micro meters"""
        if self.isEmpty():
            return None
        return self[0], self[1]

    def __str__(self) -> str:
        vs = self.getVoxelSize()
        if vs is None:
            s = "No-VoxelSize"
        else:
            s = f"{vs[0]:.2f}x{vs[1]:.2f} {MICRON_STR}/px"
        return s


# Taken from https://github.com/4DNucleome/PartSeg/blob/develop/package/PartSegImage/image_reader.py
name_to_scalar = {
    "micron": 10**-6,
    f"{MICRON_STR}": 10**-6,
    "um": 10**-6,
    "nm": 10**-9,
    "mm": 10**-3,
    "millimeter": 10**-3,
    "pm": 10**-12,
    "picometer": 100**-12,
    "nanometer": 10**-9,
    "\\u00B5m": 10**-6,
    "centimeter": 10**-2,
    "cm": 10**-2,
    "cal": 2.54 * 10**-2,
}  #: dict with known names of scalar to scalar value. Some may be  missed


def read_resolution_from_tags(
    image_file: tifffile.TiffFile,
):
    tags = image_file.pages[0].tags
    try:
        if image_file.is_imagej:
            scalar = name_to_scalar[image_file.imagej_metadata["unit"]]
        else:
            unit = tags["ResolutionUnit"].value
            if unit == 3:
                scalar = name_to_scalar["centimeter"]
            elif unit == 2:
                scalar = name_to_scalar["cal"]
            else:  # pragma: no cover
                raise KeyError(
                    f"wrong scalar {tags['ResolutionUnit']}, "
                    f"{tags['ResolutionUnit'].value}"
                )

        x_res0 = cast(float, tags["XResolution"].value[0])
        y_res0 = cast(float, tags["YResolution"].value[0])
        x_res1 = cast(float, tags["XResolution"].value[1])
        y_res1 = cast(float, tags["YResolution"].value[1])
        x_spacing = x_res1 / x_res0 * scalar
        y_spacing = y_res1 / y_res0 * scalar
    except (KeyError, ZeroDivisionError):
        x_spacing, y_spacing = None, None
    return x_spacing, y_spacing


def read_imagej_metadata(image_file: tifffile.TiffFile):
    try:
        z_spacing = (
            image_file.imagej_metadata["spacing"]
            * name_to_scalar[image_file.imagej_metadata["unit"]]
        )
    except KeyError:
        z_spacing = None
    x_spacing, y_spacing = read_resolution_from_tags(image_file)
    if x_spacing != y_spacing:
        x_spacing = None

    return x_spacing, z_spacing


class Image(FluoObject):
    """Represents an image object"""

    @classmethod
    def metadata_from_filename(cls, filename: str):
        with tifffile.TiffFile(filename) as tif:
            if tif.is_ome:
                ome = from_xml(tif.pages[0].tags["ImageDescription"].value)
                pixels = ome.images[0].pixels
                voxel_sizes: dict[str, float] = {}
                for d in "xyz":
                    attr_d = f"physical_size_{d}_quantity"
                    px_size: pint.Quantity = getattr(pixels, attr_d, None)
                    try:
                        voxel_sizes[d] = px_size.to("um").magnitude if px_size else None
                    except pint.errors.DimensionalityError:
                        voxel_sizes[d] = None
                assert voxel_sizes["x"] == voxel_sizes["y"]
                voxel_size = voxel_sizes["x"], voxel_sizes["z"]
                image_dim = (pixels.size_x, pixels.size_y, pixels.size_z)
                num_channels = pixels.size_c
                if num_channels == 1 and pixels.size_t > 1:
                    num_channels = pixels.size_t
            else:
                tags = tif.pages[0].tags
                size_x, size_y = tags["ImageWidth"].value, tags["ImageLength"].value
                # Read resolution from tags
                if tif.is_imagej:
                    xy_spacing, z_spacing = read_imagej_metadata(tif)
                    num_channels = tif.imagej_metadata.get("channels", 1)
                    if "slices" in tif.imagej_metadata:
                        size_z = tif.imagej_metadata["slices"]
                    elif "frames" in tif.imagej_metadata:
                        size_z = tif.imagej_metadata["frames"]
                    else:
                        size_z = None
                else:
                    xy_spacing, _ = read_resolution_from_tags(tif)
                    z_spacing = None
                    num_channels = None
                    size_z = len(tif.pages)

                voxel_size = (xy_spacing, z_spacing)
                image_dim = (size_x, size_y, size_z)

        return dict(
            voxel_size=voxel_size,
            image_dim=image_dim,
            num_channels=num_channels,
        )

    @classmethod
    def from_data(cls, data: np.ndarray, filename: str, **kwargs):
        """Create an Image from a ndarray. Will write the data to filename.
        data: ndarray of shape (C, Z, Y, X)
        filename: path to file to write, should end with .ome.tiff extension
        kwargs: voxel_size, image_dim
        """
        if "image_dim" in kwargs and kwargs.get("image_dim")[::-1] != data.shape[1:]:
            raise ValueError(
                f"Data shape {data.shape[1:]} != {kwargs.get('image_dim')[::-1]}"
            )
        kwargs["image_dim"] = data.shape[:0:-1]
        if "num_channels" in kwargs and kwargs.get("num_channels") != data.shape[0]:
            raise ValueError(
                f"num_channels doesn't match data shape: {data.shape=} "
                f"and num_channels={kwargs.get('num_channels')}"
            )
        kwargs["num_channels"] = data.shape[0]
        if not filename.endswith(".ome.tiff"):
            raise ValueError(
                f"Filename should have .ome.tiff extension, got {filename}"
            )
        metadata = {"axes": "CZYX"}
        if "voxel_size" in kwargs:
            micron_str = UnitsLength.MICROMETER.value
            metadata.update(
                {
                    "PhysicalSizeX": kwargs.get("voxel_size")[0],
                    "PhysicalSizeY": kwargs.get("voxel_size")[0],
                    "PhysicalSizeZ": kwargs.get("voxel_size")[1],
                    "PhysicalSizeXUnit": micron_str,
                    "PhysicalSizeYUnit": micron_str,
                    "PhysicalSizeZUnit": micron_str,
                }
            )
        tifffile.imwrite(filename, data, metadata=metadata)
        return cls(filename=filename, **kwargs)

    @classmethod
    def from_filename(cls, filename: str, **kwargs):
        """Create an Image from a file. If metadata is found in the file,
        will retrieve infos from it, like voxel size, image dim and num channels.
        kwargs have higher priority than found metadata.
        """
        metadata = cls.metadata_from_filename(filename)
        metadata.update(kwargs)  # kwargs have higher priority
        return cls(filename=filename, **metadata)

    def __init__(
        self,
        filename: str | None = None,
        voxel_size: tuple[float, float] | None = None,
        transform: np.ndarray[Any, np.float64] | None = None,
        image_dim: tuple[int, int, int] | None = None,
        num_channels: int | None = None,
        **kwargs,
    ) -> None:
        """
         Params:
        :param location: Could be a valid location: (index, filename)
        or  filename
        """
        FluoObject.__init__(self, **kwargs)
        if filename and not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found")
        self._filename: String = String(filename)
        self._voxelSize: VoxelSize = (
            VoxelSize(*voxel_size) if voxel_size else VoxelSize()
        )
        # _transform property will store the transformation matrix
        # this matrix can be used for 2D/3D alignment or
        # to represent projection directions
        self._transform: Transform = Transform(transform)
        # default origin by default is box center =
        # (Xdim/2, Ydim/2,Zdim/2)*sampling
        # origin stores a matrix that using as input the point (0,0,0)
        # provides  the position of the actual origin in the system of
        # coordinates of the default origin.
        # _origin is an object of the class Transform shifts
        # units are A.
        self._imageDim: ImageDim = ImageDim(*image_dim) if image_dim else ImageDim()
        self._num_channels: Integer = Integer(num_channels)

    def getData(self):
        """Returns data in (C, Z, Y, X) shape"""
        fname = self._filename.get()
        if fname:
            data = tifffile.imread(fname)
            if data.ndim == 2:
                ret = data[None, None]
            elif data.ndim == 3:
                if self.getNumChannels() and self.getNumChannels() > 1:
                    ret = data[:, None]
                else:
                    ret = data[None, :]
            elif data.ndim == 4:
                if self.getNumChannels() and data.shape[0] == self.getNumChannels():
                    ret = data
                else:
                    ret = np.transpose(data, (1, 0, 2, 3))
            assert ret.shape == (
                self.getNumChannels(),
                *self.getDim()[::-1],
            ), "getData failed: expected shape "
            f"{(self.getNumChannels(), *self.getDim()[::-1])}, got {ret.shape}"
            return ret

    def export(self, filename: str, channel: int | None = None, isotropic=False):
        """export the Image to the disk"""
        data = None
        vs = None
        if isotropic:
            if self.getVoxelSize() and self.getVoxelSize()[0] != self.getVoxelSize()[1]:
                data, vs = self.getDataIsotropic()
        if channel and self.getNumChannels() > 1:
            # read channel
            if data is None:
                data = self.getData()
            data = data[channel]

        if data is not None:
            tifffile.imwrite(filename, data)
        else:
            os.symlink(os.path.abspath(self.getFileName()), filename)
        return True, vs

    def getDataIsotropic(self):
        """Returns data in (C, Z, Y, X) shape
        with anisotropy corrected"""
        data = self.getData()
        vs = self.getVoxelSize()
        iso_vs = None
        if vs:
            vs_xy, vs_z = vs
            if vs_xy != vs_z:
                iso_vs = min(vs_xy, vs_z)
                data = zoom(
                    data, (1, vs_z / iso_vs, vs_xy / iso_vs, vs_xy / iso_vs), order=1
                )
        return data, iso_vs

    def isEmpty(self):
        return self.getFileName() is None

    def getVoxelSize(self) -> tuple[float, float] | None:
        """Return image voxel size. (um/pix)"""
        return self._voxelSize.getVoxelSize()

    def setVoxelSize(self, voxel_size: tuple[float, float]) -> None:
        self._voxelSize.setVoxelSize(*voxel_size)

    def getFormat(self):
        pass

    def getDataType(self):
        return self.img.dtype

    def getDimensions(self) -> tuple[int, int, int] | None:
        """getDimensions is redundant here but not in setOfImages
        create it makes easier to create protocols for both images
        and sets of images
        """
        return self.getDim()

    def getDim(self) -> tuple[int, int, int] | None:
        """Return image dimensions as tuple: (Xdim, Ydim, Zdim)"""
        x, y, z = self._imageDim.getX(), self._imageDim.getY(), self._imageDim.getZ()
        if (x is None) or (y is None) or (z is None):
            return None
        return x, y, z

    def getXDim(self) -> int | None:
        if self._imageDim.getX() is None:
            return None
        return self._imageDim.getX()

    def getYDim(self) -> int | None:
        if self._imageDim.getY() is None:
            return None
        return self._imageDim.getY()

    def getNumChannels(self) -> int | None:
        c = self._num_channels.get()
        if c:
            return c
        else:
            return 1

    def getFileName(self) -> str | None:
        """Use the _objValue attribute to store filename."""
        fname = self._filename.get()
        if fname is None:
            return None
        return fname

    def setFileName(self, filename: str) -> None:
        """Use the _objValue attribute to store filename."""
        self._filename.set(filename)

    def getBaseName(self) -> str:
        return os.path.basename(self.getFileName())

    def copyInfo(self, other: Image) -> None:
        """Copy basic information"""
        self.copyAttributes(other, "_voxelSize")

    def copyFilename(self, other: Image) -> None:
        """Copy location index and filename from other image."""
        self.setFileName(other.getFileName())

    def hasTransform(self) -> bool:
        return self._transform is not None

    def getTransform(self) -> Transform:
        return self._transform

    def setTransform(self, newTransform: Transform) -> None:
        self._transform = newTransform

    def hasOrigin(self) -> bool:
        return self._origin is not None

    def getOrigin(self) -> Transform:
        """shifts in A"""
        if self.hasOrigin():
            return self._origin
        else:
            return self._getDefaultOrigin()

    def _getDefaultOrigin(self) -> Transform:
        voxel_size = self.getVoxelSize()
        dim = self.getDim()
        if voxel_size is None or dim is None:
            return Transform()
        t = Transform()
        x, y, z = dim
        t.setShifts(
            float(x) / -2.0 * voxel_size[0],
            float(y) / -2.0 * voxel_size[0],
            float(z) * voxel_size[1],
        )
        return t  # The identity matrix

    def getShiftsFromOrigin(self) -> tuple[float, float, float]:
        origin = self.getOrigin().getShifts()
        x = origin[0]
        y = origin[1]
        z = origin[2]
        return x, y, z
        # x, y, z are floats in nanometers

    def setShiftsInOrigin(self, x: float, y: float, z: float) -> None:
        origin = self.getOrigin()
        origin.setShifts(x, y, z)

    def setOrigin(self, newOrigin: Transform) -> None:
        """If None, default origin will be set.
        Note: shifts are in nanometers"""
        if newOrigin:
            self._origin = newOrigin
        else:
            self._origin = self._getDefaultOrigin()

    def originResampled(
        self, originNotResampled: Transform, old_voxel_size: VoxelSize
    ) -> Transform | None:
        if self.getVoxelSize() is None or old_voxel_size.getVoxelSize() is None:
            raise RuntimeError("Voxel size is None")
        factor = np.array(self.getVoxelSize()) / np.array(old_voxel_size.getVoxelSize())
        shifts = originNotResampled.getShifts()
        origin = self.getOrigin()
        origin.setShifts(
            shifts[0] * factor[0], shifts[1] * factor[0], shifts[2] * factor[1]
        )
        return origin

    def __str__(self) -> str:
        """String representation of an Image."""
        return "{} ({}, {}, {} channel(s))".format(
            self.getClassName(),
            str(self._imageDim),
            str(self._voxelSize),
            str(self._num_channels),
        )

    def getFiles(self) -> set:
        return set([self.getFileName()])


class PSFModel(Image):
    """Represents a PSF.
    It is an Image but it is useful to differentiate inputs/outputs."""


class FluoImage(Image):
    """Represents a fluo object"""

    IMG_ID_FIELD = "_imgId"

    def __init__(self, **kwargs) -> None:
        """
         Params:
        :param location: Could be a valid location: (index, filename)
        or  filename
        """
        Image.__init__(self, **kwargs)
        # Image location is composed by an index and a filename
        self._psfModel: PSFModel | None = None
        self._imgId: String = String(kwargs.get("imgId", None))

    def getImgId(self) -> str | None:
        """Get unique image ID, usually retrieved from the
        file pattern provided by the user at the import time.
        """
        return self._imgId.get()

    def setImgId(self, value: str) -> None:
        self._imgId.set(value)

    def hasPSF(self) -> bool:
        return self._psfModel is not None

    def getPSF(self) -> PSFModel | None:
        """Return the PSF model"""
        return self._psfModel

    def setPSF(self, newPSF: PSFModel) -> None:
        self._psfModel = newPSF


class Coordinate3D(FluoObject):
    """This class holds the (x,y,z) position and other information
    associated with a coordinate"""

    IMAGE_ID_ATTR: str = FluoImage.IMG_ID_FIELD

    def __init__(self, **kwargs) -> None:
        FluoObject.__init__(self, **kwargs)
        self._boxSize: CsvList = CsvList(pType=float)
        self._imagePointer: Pointer = Pointer(objDoStore=False)  # points to a FluoImage
        self._transform: Transform = Transform()
        self._groupId: Integer = Integer(
            0
        )  # This may refer to a mesh, ROI, vesicle or any group of coordinates
        self._imgId = String(
            kwargs.get("imageId", None)
        )  # Used to access to the corresponding image from each coord (it's the tsId)

    def setDim(self, w: float, h: float, d: float):
        if not self._boxSize.isEmpty():
            self._boxSize.clear()
        self._boxSize.append(w)
        self._boxSize.append(h)
        self._boxSize.append(d)

    def getDim(self) -> tuple[float, float, float] | None:
        """Return the dimensions of the first image in the set."""
        if self._boxSize.isEmpty():
            return None
        w, h, d = self._boxSize[0], self._boxSize[1], self._boxSize[2]
        if (w is None) or (h is None) or (d is None):
            return None
        return w, h, d

    def setPosition(self, x: float, y: float, z: float) -> None:
        self._transform.setShifts(x, y, z)

    def getPosition(self) -> NDArray[np.float64]:
        return self._transform.getShifts()

    def setMatrix(
        self, matrix: NDArray[np.float64], convention: str | None = None
    ) -> None:
        self._transform.setMatrix(matrix)

    def getMatrix(self, convention: str | None = None) -> NDArray[np.float64]:
        return self._transform.getMatrix()

    def eulerAngles(self) -> NDArray[np.float64]:
        R = self.getMatrix()
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])

        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def getFluoImage(self) -> FluoImage | None:
        """Return the tomogram object to which
        this coordinate is associated.
        """
        return self._imagePointer.get()

    def setFluoImage(self, image: FluoImage) -> None:
        """Set the micrograph to which this coordinate belongs."""
        self._imagePointer.set(image)

    def getImageId(self) -> str:
        return self._imgId.get()

    def setImageId(self, imageId) -> None:
        self._imgId.set(imageId)

    def getImageName(self) -> str | None:
        im = self.getFluoImage()
        if im is None:
            return None
        return im.getFileName()

    def getGroupId(self) -> int:
        return self._groupId.get()

    def setGroupId(self, groupId) -> None:
        self._groupId.set(groupId)

    def hasGroupId(self) -> bool:
        return self._groupId is not None

    def __str__(self) -> str:
        return (
            f"Coordinate3D: "
            f"<{str(self._imagePointer)}, "
            f"{self._imgId}, "
            f"{self._groupId}, "
            f"{self._transform}>"
        )


class Particle(FluoImage):
    """The coordinate associated to each particle is not scaled.
    To do that, the coordinates and the particles voxel sizes should
    be compared (because of how the extraction protocol works).
    But when shifts are applied to the coordinates, it has to be considered
    that if we're operating with coordinates coming from particles,
    those shifts will be scaled, but if the coordinates come from coordinates,
    they won't be.
    """

    IMAGE_NAME_FIELD = "_imageName"
    COORD_VOL_NAME_FIELD = "_coordinate.%s" % Coordinate3D.IMAGE_ID_ATTR

    def __init__(self, **kwargs) -> None:
        FluoImage.__init__(self, **kwargs)
        # This coordinate is NOT SCALED.
        # To do that, the coordinates and subtomograms voxel sizes
        # should be compared (because of how the extraction protocol works)
        self._coordinate: Coordinate3D | None = None
        self._imageName: String = String()

    def hasCoordinate3D(self) -> bool:
        return self._coordinate is not None

    def setCoordinate3D(self, coordinate: Coordinate3D) -> None:
        self._coordinate = coordinate

    def getCoordinate3D(self) -> Coordinate3D | None:
        """Since the object Coordinate3D needs a volume,
        use the information stored in the SubTomogram to
        reconstruct the corresponding Tomogram associated to its Coordinate3D
        """
        return self._coordinate

    def getImageName(self) -> String | None:
        """Return the tomogram filename if the coordinate is not None.
        or have set the _imageName property.
        """
        if self._imageName.hasValue():
            return self._imageName.get()
        return None

    def setImageName(self, imageName: str) -> None:
        self._imageName.set(imageName)


class FluoSet(Set, FluoObject):
    _classesDict = None

    def _loadClassesDict(self) -> dict:
        if self._classesDict is None:
            from pyworkflow.plugin import Domain  # type: ignore

            self._classesDict = Domain.getObjects()
            self._classesDict.update(globals())

        return self._classesDict

    def copyInfo(self, other: FluoSet) -> None:
        """Define a dummy copyInfo function to be used
        for some generic operations on sets.
        """
        pass

    def clone(self):
        """Override the clone defined in Object
        to avoid copying _mapperPath property
        """
        pass

    def copyItems(
        self,
        otherSet: FluoSet,
        updateItemCallback: Callable[[Object, Any | None], Any] | None = None,
        itemDataIterator: Iterator | None = None,
        copyDisabledItems: bool = False,
        doClone: bool = True,
    ) -> None:
        """Copy items from another set, allowing to update items information
        based on another source of data, paired with each item.

        Params:
            otherSet: input set from where the items will be copied.
            updateItemCallback: if not None, this will be called for each item
                and each data row (if the itemDataIterator is not None). Apart
                from setting item values or new attributes, it is possible to
                set the special flag _appendItem to False, and then this item
                will not be added to the set.
            itemDataIterator: if not None, it must be an iterator that have one
                data item for each item in the set. Usually the data item is a
                data row, coming from a table stored in text files (e.g STAR)
            copyDisabledItems: By default, disabled items are not copied from the other
                set. If copyDisable=True, then the enabled property of the item
                will be ignored.
            doClone: By default, the new item that will be inserted is a "clone"
                of the input item. By using doClone=False, the same input item
                will be passed to the callback and added to the set. This will
                avoid the clone operation and the related overhead.
        """
        itemDataIter = itemDataIterator  # shortcut

        for item in otherSet:
            # copy items if enabled or copyDisabledItems=True
            if copyDisabledItems or item.isEnabled():
                newItem = item.clone() if doClone else item
                if updateItemCallback:
                    row = None if itemDataIter is None else next(itemDataIter)
                    updateItemCallback(newItem, row)
                # If updateCallBack function returns attribute
                # _appendItem to False do not append the item
                if getattr(newItem, "_appendItem", True):
                    self.append(newItem)
            else:
                if itemDataIter is not None:
                    next(itemDataIter)  # just skip disabled data row

    @classmethod
    def create(
        cls,
        outputPath: str,
        prefix: str | None = None,
        suffix: str | None = None,
        ext: str | None = None,
        **kwargs,
    ) -> FluoSet:
        """Create an empty set from the current Set class.
        Params:
           outputPath: where the output file will be written.
           prefix: prefix of the created file, if None, it will be deduced
               from the ClassName.
           suffix: additional suffix that will be added to the prefix with a
               "_" in between.
           ext: extension of the output file, be default will use .sqlite
        """
        fn = prefix or cls.__name__.lower().replace("setof", "")

        if suffix:
            fn += "_%s" % suffix

        if ext and ext[0] == ".":
            ext = ext[1:]
        fn += ".%s" % (ext or "sqlite")

        setPath = os.path.join(outputPath, fn)
        pwutils.cleanPath(setPath)

        return cls(filename=setPath, **kwargs)

    def createCopy(
        self,
        outputPath: str,
        prefix: str | None = None,
        suffix: str | None = None,
        ext: str | None = None,
        copyInfo: bool = False,
        copyItems: bool = False,
    ) -> FluoSet:
        """Make a copy of the current set to another location (e.g file).
        Params:
            outputPath: where the output file will be written.
            prefix: prefix of the created file, if None, it will be deduced
                from the ClassName.
            suffix: additional suffix that will be added to the prefix with a
                "_" in between.
            ext: extension of the output file, be default will use the same
                extension of this set filename.
            copyInfo: if True the copyInfo will be called after set creation.
            copyItems: if True the copyItems function will be called.
        """
        setObj = self.create(
            outputPath,
            prefix=prefix,
            suffix=suffix,
            ext=ext or pwutils.getExt(self.getFileName()),
        )

        if copyInfo:
            setObj.copyInfo(self)

        if copyItems:
            setObj.copyItems(self)

        return setObj

    def getFiles(self) -> set:
        return Set.getFiles(self)

    def iterItems(
        self, orderBy="id", direction="ASC", where=None, limit=None, iterate=False
    ) -> Iterable[Object]:
        return iter(Set.iterItems(self, orderBy, direction, where, limit, iterate))


class SetOfImages(FluoSet):
    """Represents a set of Images"""

    ITEM_TYPE: Object = Image

    def __init__(self, **kwargs):
        Set.__init__(self, **kwargs)
        self._voxelSize = VoxelSize()
        self._dim = ImageDim()  # Dimensions of the first image
        self._num_channels = Integer()

    def append(self, image: Image) -> None:
        """Add a image to the set."""
        if image.isEmpty():
            raise ValueError(f"Image {image} is empty!")
        # If the voxel size was set before, the same value
        # will be set for each image added to the set
        vs = self.getVoxelSize()
        im_vs = image.getVoxelSize()
        if (vs is not None) and (im_vs is None):
            image.setVoxelSize(vs)
        elif (vs is not None) and (im_vs is not None):
            if vs != im_vs:
                raise ValueError(
                    f"{image} has different voxel size than {self}, "
                    f"found {vs} and {im_vs}"
                )
        elif (vs is None) and (im_vs is not None):
            self.setVoxelSize(im_vs)
        else:
            pass

        dim = self.getDim()
        im_dim = image.getDim()
        if (dim is not None) and (im_dim is not None):
            if dim != im_dim:  # if dims are different accross images
                print(
                    f"{image} has different dimensions than {self}: {dim} and {im_dim}"
                )
                self.setDim(None)
        elif (dim is None) and (im_dim is not None):
            self.setDim(im_dim)
        else:  # im_dim is None
            raise ValueError(f"{image} has no dimension")

        c = self.getNumChannels()
        im_c = image.getNumChannels()
        if (c is not None) and (im_c is not None):
            if c != im_c:  # if num_channels are different accross images
                print(f"{image} has different channels than {self}: {c} and {im_c}")
                self.setNumChannels(None)
        elif (c is None) and (im_c is not None):
            self.setNumChannels(im_c)

        Set.append(self, image)

    def getNumChannels(self) -> int | None:
        c = self._num_channels.get()
        if c:
            return c
        else:
            return None

    def setNumChannels(self, c: int) -> None:
        self._num_channels.set(c)

    def setDim(self, dim: tuple[int, int, int] | None) -> None:
        """Store dimensions.
        This function should be called only once, to avoid reading
        dimension from image file."""
        self._dim.set_(dim)

    def copyInfo(self, other: SetOfImages) -> None:
        """Copy basic information (voxel size and psf)
        from other set of images to current one"""
        self.copyAttributes(other, "_voxelSize")

    def getFiles(self) -> set:
        filePaths = set()
        uniqueFiles = self.aggregate(["count"], "_filename", ["_filename"])

        for row in uniqueFiles:
            filePaths.add(row["_filename"])
        return filePaths

    def setDownsample(self, downFactor: float) -> None:
        """Update the values of voxelSize and scannedPixelSize
        after applying a downsampling factor of downFactor.
        """
        vs = self.getVoxelSize()
        if vs is None:
            raise RuntimeError("Couldn't downsample, voxel size is not set")
        self.setVoxelSize((vs[0] * downFactor, vs[1] * downFactor))

    def setVoxelSize(self, voxelSize: tuple[float, float]) -> None:
        """Set the voxel size and adjust the scannedPixelSize."""
        self._voxelSize.setVoxelSize(*voxelSize)

    def getVoxelSize(self) -> tuple[float, float] | None:
        return self._voxelSize.getVoxelSize()

    def writeSet(self, applyTransform: bool = False) -> None:
        for img in self:
            img.save(apply_transform=applyTransform)

    @classmethod
    def create_image(cls, filename):
        return cls.ITEM_TYPE(data=filename)

    def readSet(self, files: list[str]) -> None:
        """Populate the set with the images in the stack"""
        for i in range(len(files)):
            img = self.create_image(files[i])
            self.append(img)

    def getDim(self) -> tuple[int, int, int] | None:
        """Return the dimensions of the first image in the set."""
        if self._dim.isEmpty():
            return None
        dims = self._dim
        x, y, z = dims.getX(), dims.getY(), dims.getZ()
        if (x is None) or (y is None) or (z is None):
            return None
        return x, y, z

    def getDimensions(self) -> tuple[int, int, int] | None:
        """Return first image dimensions as a tuple: (xdim, ydim, zdim)"""
        return self.getFirstItem().getDim()

    def getMaxDataSize(self):
        """returns the maximum dimension of the set in um"""
        vs_xy, vs_z = self.getVoxelSize()
        max_set = 0
        for im in self:
            im: Image
            dx, dy, dz = im.getDimensions()
            max_im = max(dx * vs_xy, dy * vs_xy, dz * vs_z)
            max_set = max(max_set, max_im)
        return max_set

    def __str__(self) -> str:
        """String representation of a set of images."""
        s = "%s (%d items, %s, %s, %s%s)" % (
            self.getClassName(),
            self.getSize(),
            self._dimStr(),
            self._channelsStr(),
            self._voxelSizeStr(),
            self._appendStreamState(),
        )
        return s

    def _voxelSizeStr(self) -> str:
        """Returns how the voxel size is presented in a 'str' context."""
        voxel_size = self.getVoxelSize()

        if not voxel_size:
            return "No pixel size"

        return f"{voxel_size[0]:.2f}x{voxel_size[1]:.2f} {MICRON_STR}/px"

    def _channelsStr(self):
        c = self.getNumChannels()
        if c is None:
            c = "?"
        return f"{c} channel" if c == 1 else f"{c} channels"

    def _dimStr(self) -> str:
        """Return the string representing the dimensions."""
        return str(self._dim)

    def appendFromImages(self, imagesSet: SetOfImages) -> None:
        """Iterate over the images and append
        every image that is enabled.
        """
        for img in imagesSet:
            if img.isEnabled():
                self.append(img)

    def appendFromClasses(self, classesSet):
        """Iterate over the classes and the element inside each
        class and append to the set all that are enabled.
        """
        for cls in classesSet:
            if cls.isEnabled() and cls.getSize() > 0:
                for img in cls:
                    if img.isEnabled():
                        self.append(img)


class SetOfFluoImages(SetOfImages):
    """Represents a set of fluo images"""

    ITEM_TYPE = FluoImage
    REP_TYPE = FluoImage
    EXPOSE_ITEMS = True

    def __init__(self, *args, **kwargs) -> None:
        SetOfImages.__init__(self, **kwargs)
        self._psf: PSFModel | None = None

    def hasPSF(self) -> bool:
        return self._psf is not None

    def append(self, image: FluoImage) -> None:
        """Add a fluo image to the set."""
        if image.isEmpty():
            raise ValueError(f"Image {image} is empty!")

        if image.hasPSF() and self.hasPSF():
            if self._psf != image.getPSF():
                raise ValueError(
                    f"Image {image} PSF does not match with {self}'s PSF, "
                    f"found {image.getPSF()} and {self._psf}!"
                )
        elif image.hasPSF() and not self.hasPSF():
            self._psf = image.getPSF()

        SetOfImages.append(self, image)


class SetOfCoordinates3D(FluoSet):
    """Encapsulate the logic of a set of volumes coordinates.
    Each coordinate has a (x,y,z) position and is related to a Volume
    """

    ITEM_TYPE = Coordinate3D

    def __init__(self, **kwargs) -> None:
        FluoSet.__init__(self, **kwargs)
        self._boxSize: Scalar = Scalar()
        self._voxelSize: VoxelSize = VoxelSize()
        self._precedentsPointer: Pointer = (
            Pointer()
        )  # Points to the SetOfFluoImages associated to
        self._images: dict[str, FluoImage] | None = None

    def getBoxSize(self) -> float:
        """Return the box size of the particles."""
        bs = self._boxSize.get()
        return float(bs) if bs else bs

    def setBoxSize(self, boxSize: float) -> None:
        """Set the box size of the particles."""
        self._boxSize.set(boxSize)

    def getMaxBoxSize(self):
        """Return the box size in um that can contain all particles."""
        vs = self.getVoxelSize()
        if vs:
            vs_xy, vs_z = vs
        else:
            vs_xy, vs_z = 1.0, 1.0
        max_box_size_xy, max_box_size_z = None, None
        for coord in self.iterItems():
            coord: Coordinate3D
            dim = coord.getDim()
            if dim:
                dim_px = (dim[0] / vs_xy, dim[1] / vs_xy, dim[2] / vs_z)
                if (not max_box_size_xy) or max(dim_px[:2]) > max_box_size_xy:
                    max_box_size_xy = max(dim_px[:2])
                if (not max_box_size_z) or dim_px[2] > max_box_size_z:
                    max_box_size_z = dim_px[2]
        return max_box_size_xy, max_box_size_z

    def getMinBoxSize(self):
        """Return the box size in um that can contain all particles."""
        vs = self.getVoxelSize()
        if vs:
            vs_xy, vs_z = vs
        else:
            vs_xy, vs_z = 1.0, 1.0
        min_box_size_xy, min_box_size_z = None, None
        for coord in self.iterItems():
            coord: Coordinate3D
            dim = coord.getDim()
            if dim:
                dim_px = (dim[0] / vs_xy, dim[1] / vs_xy, dim[2] / vs_z)
                if (not min_box_size_xy) or min(dim_px[:2]) < min_box_size_xy:
                    min_box_size_xy = min(dim_px[:2])
                if (not min_box_size_z) or dim_px[2] < min_box_size_z:
                    min_box_size_z = dim_px[2]
        return min_box_size_xy, min_box_size_z

    def getVoxelSize(self) -> tuple[float, float] | None:
        """Return the voxel size of the particles."""
        return self._voxelSize.getVoxelSize()

    def setVoxelSize(self, voxel_size: tuple[float, float]) -> None:
        """Set the voxel size of the particles."""
        self._voxelSize.setVoxelSize(*voxel_size)

    def iterImages(self) -> Iterable[FluoImage]:
        """Iterate over the objects set associated with this
        set of coordinates.
        """
        return iter(self.getPrecedents())

    def iterImageCoordinates(self, image: FluoImage):
        """Iterates over the set of coordinates belonging to that micrograph."""
        pass

    def iterCoordinates(
        self, image: FluoImage | None = None, orderBy: str = "id"
    ) -> Iterable[Coordinate3D]:
        """Iterate over the coordinates associated with an image.
        If image=None, the iteration is performed over the whole
        set of coordinates.

        IMPORTANT NOTE: During the storing process in the database,
        Coordinates3D will lose their pointer to ther associated FluoImage.
        This method overcomes this problem by retrieving and
        relinking the FluoImage as if nothing would ever happened.

        It is recommended to use this method when working with Coordinates3D,
        being the common "iterItems" deprecated for this set.

        Example:

            >>> for coord in coordSet.iterItems()
            >>>     print(coord.getVolName())
            >>>     Error: Tomogram associated to Coordinate3D is NoneType
            >>>            (pointer lost)
            >>> for coord in coordSet.iterCoordinates()
            >>>     print(coord.getVolName())
            >>>     '/path/to/Tomo.file' retrieved correctly

        """

        # Iterate over all coordinates if imgId is None,
        # otherwise use imgId to filter the where selection
        if image is None:
            coordWhere = "1"
        elif isinstance(image, FluoImage):
            coordWhere = '%s="%s"' % ("_imgId", image.getImgId())
            print("Coordwhere is %s" % (coordWhere,))
        else:
            raise Exception("Invalid input image of type %s" % type(image))

        # Iterate over all coordinates if imgId is None,
        # otherwise use imgId to filter the where selection
        for coord in self.iterItems(where=coordWhere, orderBy=orderBy):
            # Associate the fluo image
            self._associateImage(coord)
            yield coord

    def _getFluoImage(self, imgId: str) -> FluoImage:
        """Returns the image from an imgId"""
        imgs = self._images
        if imgs is None:
            imgs = dict()

        images = self._getFluoImages()
        if imgId not in images.keys():
            im = self.getPrecedents()[{"_imgId": imgId}]
            imgs[imgId] = im
            return im
        else:
            return imgs[imgId]

    def _getFluoImages(self) -> dict[str, FluoImage]:
        imgs = self._images
        if imgs is None:
            imgs = dict()

        return imgs

    def getPrecedents(self) -> SetOfFluoImages:
        """Returns the SetOfFluoImages associated with
        this SetOfCoordinates"""
        return self._precedentsPointer.get()

    def getPrecedent(self, imgId):
        return self.getPrecedentsInvolved()[imgId]

    def setPrecedents(self, precedents: SetOfFluoImages | Pointer) -> None:
        """Set the images associated with this set of coordinates.
        Params:
            precedents: Either a SetOfFluoImages or a pointer to it.
        """
        if precedents.isPointer():
            self._precedentsPointer.copy(precedents)
        else:
            self._precedentsPointer.set(precedents)

    def getFiles(self) -> set:
        filePaths = set()
        filePaths.add(self.getFileName())
        return filePaths

    def getSummary(self) -> str:
        summary = []
        summary.append("Number of particles: %s" % self.getSize())
        summary.append("Particle size: %s" % self.getBoxSize())
        return "\n".join(summary)

    def copyInfo(self, other: SetOfCoordinates3D) -> None:
        """Copy basic information (id and other properties) but not _mapperPath or _size
        from other set of objects to current one.
        """
        self.setBoxSize(other.getBoxSize())
        if vs := other.getVoxelSize():
            self.setVoxelSize(vs)
        self.setPrecedents(other.getPrecedents())

    def __str__(self) -> str:
        """String representation of a set of coordinates."""
        s = "%s (%d items, %s%s)" % (
            self.getClassName(),
            self.getSize(),
            self._voxelSizeStr(),
            self._appendStreamState(),
        )

        return s

    def _voxelSizeStr(self) -> str:
        """Returns how the voxel size is presented in a 'str' context."""
        voxel_size = self.getVoxelSize()

        if not voxel_size:
            return "No pixel size"

        return (
            f"{voxel_size[0]:.2f}x{voxel_size[0]:.2f}x{voxel_size[1]:.2f} "
            f"{MICRON_STR}/px"
        )

    def getFirstItem(self) -> Coordinate3D:
        coord = FluoSet.getFirstItem(self)
        self._associateImage(coord)
        return coord

    def _associateImage(self, coord: Coordinate3D) -> None:
        coord.setFluoImage(self._getFluoImage(coord.getImageId()))

    def __getitem__(self, itemId: int) -> Coordinate3D:
        """Add a pointer to a FluoImage before returning the Coordinate3D"""
        coord = FluoSet.__getitem__(self, itemId)
        self._associateImage(coord)
        return coord

    def getPrecedentsInvolved(self) -> dict:
        """Returns a list with only the images involved in the particles.
        May differ when subsets are done."""

        uniqueTomos = self.aggregate(["count"], "_imgId", ["_imgId"])

        for row in uniqueTomos:
            imgId = row["_imgId"]
            # This should register the image in the internal _images
            self._getFluoImage(imgId)

        return self._images  # type: ignore

    def getImgIds(self):
        """Returns all the TS ID (tomoId) present in this set"""
        imgIds = self.aggregate(["MAX"], "_imgId", ["_imgId"])
        imgIds = [d["_imgId"] for d in imgIds]
        return imgIds


class SetOfParticles(SetOfFluoImages):
    ITEM_TYPE = Particle
    REP_TYPE = Particle
    EXPOSE_ITEMS = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._coordsPointer: Pointer = Pointer()
        self._images: dict[str, FluoImage] | None = None

    def copyInfo(self, other: SetOfParticles) -> None:
        """Copy basic information (voxel size and ctf)
        from other set of images to current one"""
        super().copyInfo(other)
        if hasattr(other, "_coordsPointer"):  # Like the vesicles in pyseg
            self.copyAttributes(other, "_coordsPointer")

    def hasCoordinates3D(self) -> bool:
        return self._coordsPointer.hasValue()

    def getCoordinates3D(self, asPointer: bool = False) -> Pointer | SetOfCoordinates3D:
        """Returns the SetOfCoordinates associated with
        this SetOfParticles"""

        return self._coordsPointer if asPointer else self._coordsPointer.get()

    def setCoordinates3D(self, coordinates: Pointer | SetOfCoordinates3D) -> None:
        """Set the SetOfCoordinates associated with
        this set of particles.
        """
        if isinstance(coordinates, Pointer):
            self._coordsPointer = coordinates
        else:
            self._coordsPointer.set(coordinates)  # FIXME: strange?

    def iterCoordinates(
        self, image: FluoImage | None = None, orderBy: str = "id"
    ) -> Iterator[Coordinate3D | None]:
        """Mimics SetOfCoordinates.iterCoordinates
        so can be passed to viewers or protocols transparently"""
        if self.hasCoordinates3D():
            for particle in self.iterParticles(image, orderBy=orderBy):
                coord = particle.getCoordinate3D()
                if coord is not None:
                    coord.setObjId(particle.getObjId())
                yield coord
        else:
            yield None

    def iterParticles(
        self, image: FluoImage | None = None, orderBy: str = "id"
    ) -> Iterator[Particle]:
        """Iterates over the particles, enriching them with the related image
        if apply so coordinate getters and setters will work
        If image=None, the iteration is performed over the whole
        set of particles.

        IMPORTANT NOTE: During the storing process in the database,
        Coordinates3D will lose their pointer to the associated Image.
        This method overcomes this problem by retrieving and
        relinking the Image as if nothing would ever happend.

        It is recommended to use this method when working with subtomograms,
        anytime you want to properly use its coordinate3D attached object.

        Example:

            >>> for subtomo in subtomos.iterItems()
            >>>     print(subtomo.getCoordinate3D().getX(SCIPION))
            >>>     Error: Tomogram associated to Coordinate3D is NoneType
            >>>            (pointer lost)
            >>> for subtomo in subtomos.iterSubtomos()
            >>>     print(subtomo.getCoordinate3D().getX(SCIPION))
            >>>     330 retrieved correctly

        """
        # Iterate over all Subtomograms if tomoId is None,
        # otherwise use tomoId to filter the where selection
        if image is None:
            particleWhere = "1"
        elif isinstance(image, FluoImage):
            particleWhere = '%s="%s"' % (
                Particle.IMAGE_NAME_FIELD,
                image.getImgId(),
            )  # TODO: add docs Particle _imageName refers to an FluoImage _imgId
        else:
            raise Exception("Invalid input tomogram of type %s" % type(image))

        for particle in self.iterItems(where=particleWhere, orderBy=orderBy):
            if particle.hasCoordinate3D():
                particle.getCoordinate3D().setVolume(self.getFluoImage(particle))
            yield particle

    def getFluoImage(self, particle: Particle) -> FluoImage | None:
        """returns and caches the tomogram related with a subtomogram.
        If the subtomograms were imported and not associated to any tomogram,
        it returns None.
        """

        # Tomogram is stored with the coordinate data
        coord = particle.getCoordinate3D()

        # If there is no coordinate associated
        if coord is None:
            return None

        # Else, there are coordinates
        imgId = coord.getImageId()

        self.initFluoImages()
        self._images = typing.cast(Dict[str, FluoImage], self._images)

        # If tsId is not cached, save both identifiers.
        if imgId not in self._images:
            img = self.getCoordinates3D().getPrecedents()[
                {FluoImage.IMG_ID_FIELD: imgId}
            ]  # type: ignore
            self._images[imgId] = img
            return img
        else:
            return self._images[imgId]

    def initFluoImages(self):
        """Initialize internal _tomos to a dictionary if not done already"""
        if self._images is None:
            self._images = dict()
        # self._images = typing.cast(Dict[str, FluoImage], self._images)

    def getFluoImages(self) -> dict[str, FluoImage]:
        """Returns a list  with only the tomograms involved in the subtomograms.
        May differ when subsets are done."""

        imageId_attr = Particle.COORD_VOL_NAME_FIELD
        if self._images is None:
            self.initFluoImages()
            self._images = typing.cast(Dict[str, FluoImage], self._images)
            uniqueImages = self.aggregate(["count"], imageId_attr, [imageId_attr])
            for row in uniqueImages:
                imgId = row[imageId_attr]
                self._images[imgId] = self.getCoordinates3D().getPrecedents()[
                    {FluoImage.IMG_ID_FIELD: imgId}
                ]  # type: ignore

        return self._images

    def append(self, particle: Particle):
        dim = self.getDim()
        im_dim = particle.getDim()
        if im_dim is None:
            raise ValueError(f"Particle {particle} dimension is None.")
        if dim is None:
            self.setDim(im_dim)
        SetOfImages.append(self, particle)


class AverageParticle(Particle):
    """Represents a Average Particle.
    It is a Particle but it is useful to differentiate outputs."""

    def __init__(self, **kwargs):
        Particle.__init__(self, **kwargs)


__all__ = [
    AverageParticle,
    Coordinate3D,
    FluoImage,
    FluoObject,
    Image,
    Particle,
    PSFModel,
    SetOfCoordinates3D,
    SetOfFluoImages,
    SetOfImages,
    SetOfParticles,
    Transform,
]
