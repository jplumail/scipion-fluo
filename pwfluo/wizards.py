import pyworkflow.wizard as pwizard
from pyworkflow.gui import dialog
from pyworkflow.gui.form import FormWindow

from pwfluo.constants import MICRON_STR
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
            metadatas = prot.load_image_info()

            # Fill voxel size boxes
            vs = set(metadatas["voxel_size"])
            if len(vs) == 0:
                dialog.showInfo(
                    "Problem", "No image found matching the pattern.", form.root
                )
            elif len(vs) == 1:
                v = vs.pop()
                if v[0]:
                    form.setVar("vs_xy", f"{v[0]*1e6:.3f}")
                if v[1]:
                    form.setVar("vs_z", f"{v[1]*1e6:.3f}")
            else:
                msg = "Found multiple values:\n"
                for vs_ in vs:
                    msg += (
                        f"\tvoxel size: {vs_[0]*1e6:.2f}x{vs_[1]*1e6:.2f} "
                        f"({MICRON_STR})\n"
                    )
                dialog.showInfo("Voxel size", msg, form.root)

        except FileNotFoundError as e:
            dialog.showInfo(
                "File not found",
                "Metadata file with acquisition not found.\n\n %s" % e,
                form.root,
            )
