import click

from control_systems.car_hillclimbing import HillClimbingCar
from control_systems.quadcopter_altitude import QuadcopterAltitude
from control_systems.base_control_system import BaseControlSystem

@click.command()
@click.option("-s", "--control_system", default="car", help="The control system you want to run")
@click.option("-p", "--show_plots", default=True, help="Plot results")
def main(control_system: str, show_plots: bool):
    """
    Main entry point to run a control system. Choose control system with the 
    --control_system flag

    """
    input_map = {
        "car": HillClimbingCar(),
        "copter": QuadcopterAltitude(),
    }

    if not control_system in input_map.keys():
        raise click.BadParameter("--control_system must be one of [car, copter]")

    obj: BaseControlSystem = input_map[control_system]
    obj.run_control_system_torch(show_plots=show_plots, recovery_module_on=False)

if __name__ == "__main__":
    main()
