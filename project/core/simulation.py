""" "Simulation class to manage OpenMC simulations."""

from pathlib import Path
import openmc
from typing import Any


class Simulation:
    def __init__(
        self,
        materials: openmc.Materials,
        geometry: openmc.Geometry,
        settings: dict[str, Any],
        name: str = "sim",
    ):
        """Initialize the Simulation object.
        Parameters
        ----------
        materials : openmc.Materials
            The materials to be used in the simulation.
        geometry : openmc.Geometry
            The geometry to be used in the simulation.
        settings : dict of str to Any
            The settings for the simulation.
        name : str, optional
            The name of the simulation.
        """
        self.materials = materials
        self.geometry = geometry
        self.settings = settings
        self.name = name
        self.output_dir = Path(__file__).parent.parent / "results" / name

    def build(self) -> openmc.Model:
        """Build the OpenMC model.
        Returns
        -------
        openmc.Model
            The built OpenMC model.
        """
        self.settings_obj = openmc.Settings(**self.settings | {"output": {"path": str(self.output_dir)}})

        self.model = openmc.Model(
            materials=self.materials, geometry=self.geometry, settings=self.settings_obj
        )
        return self.model

    def run(self) -> Path:
        """Run the simulation.
        Returns
        -------
        Path
            The path to the output directory.
        """
        if self.model is None:
            self.build()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output: Path = self.model.run(cwd=str(self.output_dir))

        return output
