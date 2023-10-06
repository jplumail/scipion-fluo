import pint
import pyworkflow.wizard as pwizard
from pyworkflow.gui import dialog
from pyworkflow.gui.form import FormWindow

from pwfluo.protocols.protocol_base import ProtFluoImportFile, ProtFluoImportFiles


class FluoWizard(pwizard.Wizard):
    pass


class ImportAcquisitionWizard(FluoWizard):
    _targets = [
        (ProtFluoImportFiles, ["importWizard"]),
        (ProtFluoImportFile, ["importWizard"]),
    ]

    @classmethod
    def show(cls, form: FormWindow, *params):
        try:
            prot = form.protocol  # type: ProtFluoImportFiles | ProtFluoImportFile
            voxel_sizes, dims = prot.load_image_info()

            # Fill voxel size boxes
            vs_xy = list(set(voxel_sizes["x"] + voxel_sizes["y"]))
            vs_z = list(set(voxel_sizes["z"]))
            if not (len(vs_xy) == 0 and len(vs_z) == 0):
                for vs, name, attr in [(vs_xy, "XY", "vs_xy"), (vs_z, "Z", "vs_z")]:
                    if len(vs) == 0:
                        msg = f"{name} no voxel size found.\n"
                        # dialog.showInfo(f"Voxel size {name}", msg, form.root)
                    elif len(vs) > 1:
                        msg = f"{name} found multiple values:\n"
                        for vs_ in vs:
                            if isinstance(vs_, pint.Quantity):
                                msg += f"\t{name} voxel size: {vs_.to('um'):.3f}\n"
                            else:
                                msg += f"\t{name} voxel size: {vs_} (no unit)\n"
                        # dialog.showInfo(f"Voxel size {name}", msg, form.root)
                    else:
                        v = vs[0]
                        if isinstance(v, pint.Quantity):
                            v = v.to("um")
                            msg = f"{name} voxel size: {v:.3f}\n"
                            v = v.magnitude
                        else:
                            msg = f"{name} voxel size: {vs[0]} (no unit)\n"
                        msg += "\n*Do you want to use detected voxel size values?*"
                        comment = ""
                        if prot.hasAttribute(attr):
                            form.setVar(attr, v)
                        else:
                            comment += "%s = %s\n" % (attr, v)
                        if comment:
                            prot.setObjComment(comment)

            # Fill transpose T<>Z axes
            problematic_images = []
            for fname in dims:
                dim = dims[fname]
                if dim.Z == 1 and dim.T > 1:
                    problematic_images.append(fname)
            if len(problematic_images) > 0:
                msg = f"Found {'files' if len(problematic_images)>1 else 'a file'} "
                "with a temporal dimension greater than 1, "
                "and Z dimension equal to 1:\n"
                for fname in problematic_images:
                    msg += "\t" + f"{fname}: {dims[fname]}" + "\n"
                msg += "\n*Do you want to transpose the T and Z axes for "
                msg += (
                    "these images?*" if len(problematic_images) > 1 else "this image?*"
                )
                msg += "\n"
                response = dialog.askYesNo("transpose axes?", msg, form.root)
                if response:
                    form.setVar("transpose_tz", True)
                else:
                    form.setVar("transpose_tz", False)

        except FileNotFoundError as e:
            dialog.showInfo(
                "File not found",
                "Metadata file with acquisition not found.\n\n %s" % e,
                form.root,
            )
