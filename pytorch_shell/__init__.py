from .membrane_energy import membrane_energy
from .bending_energy import bending_energy
from .shell_energy import shell_energy


try:

    from skshapes.morphing.metrics import Metric
    class ShellEnergyMetric(Metric):
        def __init__(self, weight=0.001):
            self.weight = weight

        def __call__(
            self,
            points_sequence,
            velocities_sequence,
            edges=None,
            triangles=None,
        ):
            points_undef = points_sequence
            points_def = points_sequence + velocities_sequence
            energy = shell_energy(
                points_undef=points_undef,
                points_def=points_def,
                triangles=triangles,
                weight=self.weight,
            ).mean()
            return energy

    # Pipeline
    import skshapes as sks
    def apply_pipeline(source, target, pipeline, parameter=None):

        if "align" in pipeline.keys():
            for step in pipeline["align"]:

                model = step["model"]
                loss = step["loss"]
                n_iter = step["n_iter"]
                regularization = step["regularization"]

                registration = sks.Registration(
                    model = model,
                    loss = loss,
                    verbose = False,
                    n_iter=n_iter,
                    regularization = regularization,
                )
                source = registration.fit_transform(
                    source=source,
                    target=target
                )

        for step in pipeline["steps"]:

            model = step["model"]
            loss = step["loss"]
            n_iter = step["n_iter"]
            regularization = step["regularization"]

            registration = sks.Registration(
                model = step["model"],
                loss = step["loss"],
                verbose = False,
                n_iter=step["n_iter"],
                regularization = step["regularization"],
            )

            registration.fit(
                source=source,
                target=target,
                initial_parameter=parameter,
            )

            parameter = registration.parameter_

            output = step["model"].morph(
                shape=source,
                parameter=parameter,
                return_path = True,
                return_regularization = True,
            )


            path = output.path
            path_length = output.regularization
            morphed_shape = output.morphed_shape

            print(f"Path energy: {path_length:.2e}, Loss: {loss(morphed_shape, target):.2e}")

        # sks.Browser(path).show()
        return parameter
    
except ImportError:
    pass # skshapes not installed