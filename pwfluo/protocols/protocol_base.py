# **************************************************************************
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

import os
import re
from typing import List, Optional, Type, TypeVar

import pint
import pyworkflow as pw
import pyworkflow.protocol.params as params
from aicsimageio import AICSImage
from aicsimageio.dimensions import Dimensions
from pyworkflow import utils as pwutils
from pyworkflow.mapper.sqlite_db import SqliteDb
from pyworkflow.object import Set
from pyworkflow.protocol.params import (
    FloatParam,
    Form,
    PointerParam,
)
from pyworkflow.utils.path import createAbsLink, removeExt
from pyworkflow.utils.properties import Message

import pwfluo.objects as pwfluoobj
from pwfluo.protocols.import_ import ProtImport, ProtImportFile, ProtImportFiles

READERS = {}
READERS["Automatic"] = None
READERS.update({getattr(r, "__name__"): r for r in AICSImage.SUPPORTED_READERS})
READERS_KEYS = list(READERS.keys())


def _getUniqueFileName(pattern, filename, filePaths=None):
    if filePaths is None:
        filePaths = [re.split(r"[$*#?]", pattern)[0]]

    commPath = pwutils.commonPath(filePaths)
    return filename.replace(commPath + "/", "").replace("/", "_")


class ProtFluoBase:
    T = TypeVar("T", bound=Set)
    OUTPUT_PREFIX: str

    def _createSet(self, SetClass: Type[T], template, suffix, **kwargs):
        """Create a set and set the filename using the suffix.
        If the file exists, it will be deleted."""
        setFn = self._getPath(template % suffix)
        # Close the connection to the database if
        # it is open before deleting the file
        pw.utils.cleanPath(setFn)

        SqliteDb.closeConnection(setFn)
        setObj = SetClass(filename=setFn, **kwargs)
        return setObj

    def _createSetOfCoordinates3D(
        self, volSet: pwfluoobj.SetOfFluoImages, suffix: str = ""
    ):
        coord3DSet = self._createSet(
            pwfluoobj.SetOfCoordinates3D,
            "coordinates%s.sqlite",
            suffix,
            indexes=[pwfluoobj.Coordinate3D.IMAGE_ID_ATTR],
        )
        coord3DSet.setPrecedents(volSet)
        return coord3DSet

    def _createSetOfFluoImages(self, suffix: str = ""):
        return self._createSet(pwfluoobj.SetOfFluoImages, "fluoimages%s.sqlite", suffix)

    def _createSetOfParticles(self, suffix: str = ""):
        return self._createSet(pwfluoobj.SetOfParticles, "particles%s.sqlite", suffix)

    def _getOutputSuffix(self, cls: type):
        """Get the name to be used for a new output.
        For example: output3DCoordinates7.
        It should take into account previous outputs
        and number with a higher value.
        """
        maxCounter = -1
        for attrName, _ in self.iterOutputAttributes(cls):
            suffix = attrName.replace(self.OUTPUT_PREFIX, "")
            try:
                counter = int(suffix)
            except ValueError:
                counter = 1  # when there is not number assume 1
            maxCounter = max(counter, maxCounter)

        return str(maxCounter + 1) if maxCounter > 0 else ""  # empty if not output


class ProtFluoPicking(ProtImport, ProtFluoBase):
    OUTPUT_PREFIX = "output3DCoordinates"

    """ Base class for Fluo boxing protocols. """

    def _defineParams(self, form: Form) -> None:
        form.addSection(label="Input")
        form.addParam(
            "inputFluoImages",
            PointerParam,
            label="Input Images",
            important=True,
            pointerClass="SetOfFluoImages",
            help="Select the Image to be used during picking.",
        )

    def _summary(self) -> List[str]:
        summary = []
        if self.isFinished() and self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:*\n%s" % (key, output.getSummary()))
        else:
            summary.append(Message.TEXT_NO_OUTPUT_CO)
        return summary


class ProtFluoImportBase(ProtFluoBase):
    def __init__(self):
        self.images = None

    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        group = form.addGroup("Image Info")
        group.addParam(
            "importWizard",
            params.LabelParam,
            important=True,
            label="Use the wizard button to import parameters.",
            help="The wizard will try to import image parameters "
            "using the Reader above.\n"
            "If not found, required ones should be provided.",
        )
        group.addParam("vs_xy", FloatParam, label="XY (μm/px)")
        group.addParam("vs_z", FloatParam, label="Z (μm/px)")
        group.addParam(
            "transpose_tz",
            params.BooleanParam,
            default=True,
            label="transpose T<>Z axes when possible?",
            help="Will transpose T<>Z when Z=1 and T>1.",
        )

    def _defineImportParams(self, form: Form) -> None:
        form.addParam(
            "reader",
            params.EnumParam,
            choices=READERS_KEYS,
            display=params.EnumParam.DISPLAY_COMBO,
            label="Reader",
            help="Choose the reader"
            "The DefaultReader finds a reader that works for your image."
            "BioformatsReader corresponds to the ImageJ reader"
            "(requires java and maven to be installed)",
        )

    def getReader(self):
        if self.reader.get() is not None:
            return READERS[READERS_KEYS[self.reader.get()]]
        else:
            return READERS["Automatic"]

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return ""

    def _hasOutput(self):
        return self.images is not None

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* (μm/px)"
            )
        return methods


class ProtFluoImportFiles(ProtFluoImportBase, ProtImportFiles):
    T = TypeVar("T", bound=pwfluoobj.FluoImage)

    def __init__(self, **args):
        ProtImportFiles.__init__(self, **args)
        ProtFluoImportBase.__init__(self)

    def importStep(self, obj: Type[T]):
        """Copy images matching the filename pattern
        Register other parameters.
        """
        pattern = self.getPattern()
        self.info("Using pattern: '%s'" % pattern)

        if obj is pwfluoobj.FluoImage:
            imgSet = self._createSetOfFluoImages()
        elif obj is pwfluoobj.Particle:
            imgSet = self._createSetOfParticles()
        else:
            raise NotImplementedError()
        voxel_size: tuple[float, float] = self.vs_xy.get(), self.vs_z.get()
        imgSet.setVoxelSize(voxel_size)

        fileNameList = []
        for fileName, fileId in self.iterFiles():
            img = obj(data=fileName, reader=self.getReader())
            img.setVoxelSize(voxel_size)

            # Set default origin
            origin = pwfluoobj.Transform()
            dim = img.getDim()
            if dim is None:
                raise ValueError("Image '%s' has no dimension" % fileName)
            x, y, z = dim
            origin.setShifts(
                x / -2.0 * voxel_size[0],
                y / -2.0 * voxel_size[0],
                z / -2.0 * voxel_size[1],
            )
            img.setOrigin(origin)

            newFileName = os.path.basename(fileName).split(":")[0]
            if newFileName in fileNameList:
                newFileName = _getUniqueFileName(
                    self.getPattern(), fileName.split(":")[0]
                )

            fileNameList.append(newFileName)

            imgId = removeExt(newFileName)
            img.setImgId(imgId)

            createAbsLink(
                os.path.abspath(fileName),
                os.path.abspath(self._getExtraPath(newFileName)),
            )

            img.cleanObjId()
            img.setFileName(self._getExtraPath(newFileName))
            if img.img and self.transpose_tz.get():
                if img.img.dims.T > 1 and img.img.dims.Z == 1:
                    img.transposeTZ()
            imgSet.append(img)

        imgSet.write()
        self._defineOutputs(**{self.OUTPUT_NAME: imgSet})

    def load_image_info(self):
        """Return a proper acquisitionInfo (dict)
        or an error message (str).
        """
        voxel_sizes: dict[str, list[float | pint.Quantity]] = {
            "x": [],
            "y": [],
            "z": [],
        }
        dims: dict[str, Dimensions] = {}
        for fname, _ in self.iterFiles():
            im = AICSImage(fname, reader=self.getReader())
            dims[os.path.basename(fname)] = im.dims
            try:
                pixels = im.ome_metadata.images[0].pixels
                for d in "xyz":
                    attr_d = f"physical_size_{d}"
                    if getattr(pixels, attr_d + "_quantity", None):
                        voxel_sizes[d].append(
                            getattr(pixels, attr_d + "_quantity")
                        )  # returns a pint.Quantity
                    elif getattr(pixels, attr_d) is not None:
                        voxel_sizes[d].append(getattr(pixels, attr_d))  # float
            except NotImplementedError:
                pass
        for k in voxel_sizes:
            voxel_sizes[k] = list(set(voxel_sizes[k]))
        return voxel_sizes, dims

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_"
                + str(os.path.basename(fileName)).split(".")[0]
                + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(os.path.basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors


class ProtFluoImportFile(ProtFluoImportBase, ProtImportFile):
    T = TypeVar("T", bound=pwfluoobj.FluoImage)

    def __init__(self, **args):
        ProtFluoImportBase.__init__(self)
        ProtImportFile.__init__(self, **args)

    def importImageStep(self, obj: type[T]) -> None:
        """Copy the file.
        Register other parameters.
        """
        self.info("")

        file_path = self.filePath.get()
        img = obj(data=file_path, reader=self.getReader())
        voxel_size: tuple[float, float] = self.vs_xy.get(), self.vs_z.get()
        img.setVoxelSize(voxel_size)

        # Set default origin
        origin = pwfluoobj.Transform()
        dim = img.getDim()
        if dim is None:
            raise ValueError("Image '%s' has no dimension" % file_path)
        x, y, z = dim
        origin.setShifts(
            x / -2.0 * voxel_size[0],
            y / -2.0 * voxel_size[0],
            z / -2.0 * voxel_size[1],
        )
        img.setOrigin(origin)

        newFileName = os.path.basename(file_path)

        imgId = removeExt(newFileName)
        img.setImgId(imgId)

        createAbsLink(
            os.path.abspath(file_path), os.path.abspath(self._getExtraPath(newFileName))
        )

        img.cleanObjId()
        img.setFileName(self._getExtraPath(newFileName))
        if img.img and self.transpose_tz.get():
            if img.img.dims.T > 1 and img.img.dims.Z == 1:
                img.transposeTZ()

        self._defineOutputs(**{self.OUTPUT_NAME: img})

    def load_image_info(self):
        """Return a proper acquisitionInfo (dict)
        or an error message (str).
        """
        voxel_sizes: dict[str, list[float | pint.Quantity]] = {
            "x": [],
            "y": [],
            "z": [],
        }
        fname = self.filePath.get()
        dims: dict[str, Dimensions] = {}
        if fname:
            im = AICSImage(fname, reader=self.getReader())
            dims[fname] = im.dims
            try:
                pixels = im.ome_metadata.images[0].pixels
                for d in "xyz":
                    attr_d = f"physical_size_{d}"
                    if getattr(pixels, attr_d + "_quantity", None):
                        voxel_sizes[d].append(
                            getattr(pixels, attr_d + "_quantity")
                        )  # returns a pint.Quantity
                    elif getattr(pixels, attr_d) is not None:
                        voxel_sizes[d].append(getattr(pixels, attr_d))  # float
            except NotImplementedError:
                pass
        for k in voxel_sizes:
            voxel_sizes[k] = list(set(voxel_sizes[k]))
        return voxel_sizes, dims

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_"
                + str(os.path.basename(fileName)).split(".")[0]
                + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(os.path.basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors
