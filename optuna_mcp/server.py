import argparse
from dataclasses import dataclass
from importlib import import_module
import json
import sys
import threading
import typing

import kaleido
from kaleido.errors import ChromeNotFoundError
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from mcp.types import INTERNAL_ERROR
from mcp.types import INVALID_PARAMS
import optuna
import optuna_dashboard
from plotly.graph_objects import Figure
from pydantic import BaseModel
from pydantic import Field

from optuna_mcp.version import __version__


SamplerName = typing.Literal["TPESampler", "NSGAIISampler", "RandomSampler", "GPSampler"]
DirectionName = typing.Literal["minimize", "maximize"]
DEFAULT_STARTUP_TRIALS = 10
STARTUP_TRIALS_ATTR = "optuna_mcp.n_startup_trials"
OPTIMIZATION_INSTRUCTIONS = """
Before starting an optimization, check that the user has provided the objective
function or evaluation method, the direction (minimize or maximize) for each
objective, and the search space for each parameter (type and bounds or category
choices, including any relevant endpoint or step requirements). If required
information is missing or ambiguous, ask the user for it and wait for their
answer before creating a study or asking for a trial. Use the stored directions
when resuming an existing study. Ask about constraints or the stopping criterion
when they are required to carry out the request and remain unclear.
Use the default GPSampler unless the user requests a different sampler.
If the user specifies an initial trial count for GPSampler, pass it as
n_startup_trials to create_study. This also updates the count for an existing
study when requested.
With GPSampler, call ask_startup_trials after confirming the search space.
Present all returned conditions together with their trial numbers. Register
each result with tell. Once the initial trials are complete, use ask for the
next model-based condition.
""".strip()


class OptunaMCP(FastMCP):
    def __init__(
        self, name: str, storage: str | None = None, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        kwargs["name"] = name
        kwargs.setdefault("instructions", OPTIMIZATION_INSTRUCTIONS)
        super().__init__(*args, **kwargs)
        self.storage: str | optuna.storages.BaseStorage | None = storage
        self.study: optuna.study.Study | None = None
        self.dashboard_thread_port: tuple[threading.Thread, int] | None = None


@dataclass
class TrialToAdd:
    """
    A trial to be added to an Optuna study.

    Attributes:
        params: The parameter values for the trial.
        distributions: The distributions used for the parameters.
            A key is the parameter name and a value is a distribution.
            The distribution is a dictionary that can be converted to a JSON string, e.g.,
            {
                "name": "IntDistribution",
                "attributes": {"step": null, "low": 1, "high": 9, "log": false}
            }.
            The name of the distribution must be one of the following:
            - FloatDistribution
            - IntDistribution
            - CategoricalDistribution
        values: The objective values for the trial, or None if not set.
            If the state is "COMPLETE", this must be set.
        state: The state of the trial.
            - "COMPLETE": The trial completed successfully.
            - "PRUNED": The trial was pruned.
            - "FAIL": The trial failed.
        user_attrs: User-defined attributes for the trial, or None if not set.
        system_attrs: System-defined attributes for the trial, or None if not set.
    """

    params: dict[str, typing.Any]
    distributions: dict[str, typing.Any]
    values: list[float] | None
    state: typing.Literal["COMPLETE", "PRUNED", "FAIL"]
    user_attrs: dict[str, typing.Any] | None
    system_attrs: dict[str, typing.Any] | None


class StudyResponse(BaseModel):
    study_name: str
    sampler_name: SamplerName | None = Field(
        default=None, description="The name of the sampler used in the study."
    )
    directions: list[DirectionName] | None = Field(
        default=None, description="The optimization directions for each objective."
    )
    metric_names: list[str] | None = Field(
        default=None, description="The metric names for each objective."
    )
    n_startup_trials: int | None = Field(
        default=None, description="The study's configured initial trial count for GPSampler."
    )


class TrialResponse(BaseModel):
    trial_number: int
    params: dict[str, typing.Any] | None = Field(
        default=None, description="The parameter values suggested by the trial."
    )
    values: list[float] | None = Field(
        default=None, description="The objective values of the trial."
    )
    user_attrs: dict[str, typing.Any] | None = Field(
        default=None, description="User-defined attributes for the trial."
    )
    system_attrs: dict[str, typing.Any] | None = Field(
        default=None, description="System-defined attributes for the trial."
    )


def _render_png(figure: Figure) -> Image:
    try:
        data = kaleido.calc_fig_sync(
            figure,
            opts={"format": "png"},
            kopts={"enable_sandbox": True, "enable_extensions": False, "mathjax": False},
        )
    except ChromeNotFoundError as error:
        raise RuntimeError(
            "PNG export requires Chrome or Chromium. Install a supported browser "
            "or set BROWSER_PATH to its executable."
        ) from error
    return Image(data=data, format="png")


def register_tools(mcp: OptunaMCP) -> OptunaMCP:
    def startup_trial_count(study: optuna.study.Study) -> int:
        return study.user_attrs.get(STARTUP_TRIALS_ATTR, DEFAULT_STARTUP_TRIALS)

    @mcp.tool(structured_output=True)
    def create_study(
        study_name: str,
        directions: list[DirectionName] | None = None,
        n_startup_trials: typing.Annotated[int, Field(ge=1, strict=True)] | None = None,
    ) -> StudyResponse:
        """Create or load an Optuna study. The default sampler is GPSampler.

        For a new study, ask the user for any missing objective, direction, or
        search-space information before calling this tool. Provide a non-empty
        directions list with one entry per objective. Omit directions only to
        load an existing study using its stored directions.

        Set n_startup_trials to change the number of completed trials GPSampler
        needs before using its model. The new count is saved in the study.
        Omit it to keep the stored count, or use 10 for a study without one.
        """

        sampler = (
            mcp.study.sampler
            if mcp.study is not None and mcp.study.study_name == study_name
            else optuna.samplers.GPSampler(n_startup_trials=DEFAULT_STARTUP_TRIALS)
        )
        if directions is None and mcp.storage is not None:
            try:
                study = optuna.load_study(
                    study_name=study_name, storage=mcp.storage, sampler=sampler
                )
            except KeyError:
                study = None
        elif directions:
            study = optuna.create_study(
                study_name=study_name,
                storage=mcp.storage,
                load_if_exists=True,
                directions=directions,
                sampler=sampler,
            )
        else:
            study = None

        if study is None:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="Ask the user whether to minimize or maximize each objective, "
                    "then provide a non-empty directions list to create a new study.",
                )
            )
        stored_directions = [typing.cast(DirectionName, d.name.lower()) for d in study.directions]
        if directions is not None and directions != stored_directions:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"The existing study has directions {stored_directions}. Ask the user "
                    "whether to resume it or create a study with a different name and directions.",
                )
            )

        startup_trials = startup_trial_count(study)
        if n_startup_trials is not None and n_startup_trials != startup_trials:
            study.set_user_attr(STARTUP_TRIALS_ATTR, n_startup_trials)
            startup_trials = n_startup_trials
        # Optuna does not expose the configured startup threshold publicly.
        if isinstance(sampler, optuna.samplers.GPSampler) and (
            sampler._n_startup_trials != startup_trials
        ):
            study.sampler = optuna.samplers.GPSampler(n_startup_trials=startup_trials)

        mcp.study = study
        if mcp.storage is None:
            mcp.storage = study._storage

        return StudyResponse(
            study_name=study.study_name,
            sampler_name=typing.cast(SamplerName, type(study.sampler).__name__),
            directions=stored_directions,
            n_startup_trials=startup_trials,
        )

    @mcp.tool(structured_output=True)
    def get_all_study_names() -> list[StudyResponse]:
        """Get all study names from the storage."""
        storage: str | optuna.storages.BaseStorage | None = None
        if mcp.study is not None:
            storage = mcp.study._storage
        elif mcp.storage is not None:
            storage = mcp.storage
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="No storage specified."))

        study_names = optuna.get_all_study_names(storage)
        return [StudyResponse(study_name=name) for name in study_names]

    def parse_search_space(search_space: dict) -> dict[str, optuna.distributions.BaseDistribution]:
        if not search_space:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="Ask the user for the search space, including parameter types "
                    "and bounds or choices, then provide a non-empty search_space.",
                )
            )
        try:
            return {
                name: optuna.distributions.json_to_distribution(json.dumps(dist))
                for name, dist in search_space.items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Invalid search_space: {error}. Ask the user for any missing "
                    "or ambiguous parameter types, bounds, or choices before retrying.",
                )
            ) from error

    def suggest_trial(
        study: optuna.study.Study,
        distributions: dict[str, optuna.distributions.BaseDistribution],
    ) -> TrialResponse:
        trial = study.ask(fixed_distributions=distributions)
        return TrialResponse(trial_number=trial.number, params=trial.params)

    @mcp.tool(structured_output=True)
    def ask(search_space: dict) -> TrialResponse:
        """Suggest a single trial using Optuna.

        For GPSampler, use ask_startup_trials first to present the initial
        random conditions together.

        search_space must be a non-empty dictionary mapping parameter names to
        Optuna distributions. Ask the user for any missing parameter types,
        numeric bounds, or categorical choices before calling this tool.

        Example:
            {"x": {"name": "FloatDistribution", "attributes": {"step": null, "low": -10.0, "high": 10.0, "log": false}}}
        """
        distributions = parse_search_space(search_space)
        study = mcp.study
        if study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        return suggest_trial(study, distributions)

    @mcp.tool(structured_output=True)
    def ask_startup_trials(search_space: dict) -> list[TrialResponse]:
        """Return the remaining GPSampler startup conditions in one batch.

        Use the same search_space format as ask. This tool reuses pending
        trials with the same search space and creates only the additional
        trials needed to reach the startup threshold (10 by default).
        Present the whole batch with trial numbers and use tell for each
        result. An empty batch means the startup threshold has been reached.
        """
        distributions = parse_search_space(search_space)
        study = mcp.study
        if study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )
        sampler = study.sampler
        if not isinstance(sampler, optuna.samplers.GPSampler):
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message="ask_startup_trials requires GPSampler.")
            )

        complete = optuna.trial.TrialState.COMPLETE
        running = optuna.trial.TrialState.RUNNING
        trials = study.get_trials(deepcopy=False, states=(complete, running))
        # Optuna does not expose the configured startup threshold publicly.
        remaining = sampler._n_startup_trials - sum(trial.state == complete for trial in trials)
        if remaining <= 0:
            return []

        pending = [
            TrialResponse(trial_number=trial.number, params=trial.params)
            for trial in trials
            if trial.state == running and trial.distributions == distributions
        ][:remaining]
        pending.extend(
            suggest_trial(study, distributions) for _ in range(remaining - len(pending))
        )
        return pending

    @mcp.tool(structured_output=True)
    def tell(trial_number: int, values: float | list[float]) -> TrialResponse:
        """Report the result of a trial"""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        mcp.study.tell(
            trial=trial_number,
            values=values,
            state=optuna.trial.TrialState.COMPLETE,
            skip_if_finished=True,
        )
        return TrialResponse(
            trial_number=trial_number,
            values=[values] if isinstance(values, float) else values,
        )

    @mcp.tool(structured_output=True)
    def set_sampler(
        name: SamplerName,
    ) -> StudyResponse:
        """Set the sampler for the study.
        The sampler must be one of the following:
        - TPESampler
        - NSGAIISampler
        - RandomSampler
        - GPSampler

        GPSampler is the default. Use this tool to explicitly override it.
        """
        study = mcp.study
        if study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        if name == "GPSampler":
            n_startup_trials = startup_trial_count(study)
            study.sampler = optuna.samplers.GPSampler(n_startup_trials=n_startup_trials)
        else:
            n_startup_trials = None
            study.sampler = getattr(optuna.samplers, name)()
        return StudyResponse(
            study_name=study.study_name,
            sampler_name=name,
            n_startup_trials=n_startup_trials,
        )

    @mcp.tool(structured_output=True)
    def set_trial_user_attr(trial_number: int, key: str, value: typing.Any) -> str:
        """Set user attributes for a trial"""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        storage = mcp.study._storage
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            mcp.study._study_id, trial_number
        )
        storage.set_trial_user_attr(trial_id, key, value)
        return f"User attribute {key} set to {json.dumps(value)} for trial {trial_number}"

    @mcp.tool(structured_output=True)
    def get_trial_user_attrs(trial_number: int) -> TrialResponse:
        """Get user attributes in a trial"""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        storage = mcp.study._storage
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            mcp.study._study_id, trial_number
        )
        trial = storage.get_trial(trial_id)
        return TrialResponse(
            trial_number=trial_number,
            user_attrs=trial.user_attrs,
        )

    @mcp.tool(structured_output=True)
    def set_metric_names(metric_names: list[str]) -> StudyResponse:
        """Set metric_names. metric_names are labels used to distinguish what each objective value is.

        Args:
            metric_names:
                The list of metric name for each objective value.
                The length of metric_names list must be the same with the number of objectives.
        """
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )
        mcp.study.set_metric_names(metric_names)
        return StudyResponse(
            study_name=mcp.study.study_name,
            metric_names=metric_names,
        )

    @mcp.tool(structured_output=True)
    def get_metric_names() -> StudyResponse:
        """Get metric_names"""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        return StudyResponse(
            study_name=mcp.study.study_name,
            metric_names=mcp.study.metric_names,
        )

    @mcp.tool(structured_output=True)
    def get_directions() -> StudyResponse:
        """Get the directions of the study."""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        directions = [d.name.lower() for d in mcp.study.directions]
        return StudyResponse(
            study_name=mcp.study.study_name,
            directions=typing.cast(list[DirectionName], directions),
        )

    @mcp.tool(structured_output=False)
    def get_trials() -> str:
        """Get all trials in a CSV format"""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        csv_string = mcp.study.trials_dataframe().to_csv()
        return f"Trials: \n{csv_string}"

    @mcp.tool(structured_output=True)
    def best_trial() -> TrialResponse:
        """Get the best trial

        This feature can only be used for single-objective optimization. If your study is multi-objective, use best_trials instead.
        """
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        trial = mcp.study.best_trial
        return TrialResponse(
            trial_number=trial.number,
            params=trial.params,
            values=trial.values,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
        )

    @mcp.tool(structured_output=True)
    def best_trials() -> list[TrialResponse]:
        """Return trials located at the Pareto front in the study."""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        return [
            TrialResponse(
                trial_number=trial.number,
                params=trial.params,
                values=trial.values,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
            )
            for trial in mcp.study.best_trials
        ]

    def _create_trial(trial: TrialToAdd) -> optuna.trial.FrozenTrial:
        """Create a trial from the given parameters."""
        return optuna.trial.create_trial(
            params=trial.params,
            distributions={
                k: optuna.distributions.json_to_distribution(json.dumps(d))
                for k, d in trial.distributions.items()
            },
            values=trial.values,
            state=optuna.trial.TrialState[trial.state],
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
        )

    @mcp.tool(structured_output=True)
    def add_trial(trial: TrialToAdd) -> str:
        """Add a trial to the study."""
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        mcp.study.add_trial(_create_trial(trial))
        return "Trial was added."

    @mcp.tool(structured_output=True)
    def add_trials(trials: list[TrialToAdd]) -> str:
        """Add multiple trials to the study."""
        frozen_trials = [_create_trial(trial) for trial in trials]
        if mcp.study is None:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        mcp.study.add_trials(frozen_trials)
        return f"{len(trials)} trials were added."

    @mcp.tool()
    def plot_optimization_history(
        target: int | None = None,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the optimization history plot as an image.

        Args:
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
                For single-objective optimization, None (auto) is recommended.
                For multi-objective optimization, this must be specified.
            target_name:
                Target's name to display on the axis label and the legend.
        """
        fig = optuna.visualization.plot_optimization_history(
            mcp.study,
            target=(lambda t: t.values[target]) if target is not None else None,
            target_name=target_name,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_hypervolume_history(
        reference_point: list[float],
    ) -> Image:
        """Return the hypervolume history plot as an image.

        Args:
            reference_point:
                A list of reference points to calculate the hypervolume.
        """
        fig = optuna.visualization.plot_hypervolume_history(
            mcp.study,
            reference_point=reference_point,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_pareto_front(
        target_names: list[str] | None = None,
        include_dominated_trials: bool = True,
        targets: list[int] | None = None,
    ) -> Image:
        """Return the Pareto front plot as an image for multi-objective optimization.

        Args:
            target_names:
                Objective name list used as the axis titles. If :obj:`None` is specified,
                "Objective {objective_index}" is used instead. If ``targets`` is specified
                for a study that does not contain any completed trial,
                ``target_name`` must be specified.
            include_dominated_trials:
                A flag to include all dominated trial's objective values.
            targets:
                A list of indices to specify the objective values to display.
                Note that this is 0-indexed, i.e., to plot the first and second objective value, set this to [0, 1].
                If the number of objectives is neither 2 nor 3, ``targets`` must be specified.
                By default, all objectives are displayed.
        """
        fig = optuna.visualization.plot_pareto_front(
            mcp.study,
            target_names=target_names,
            include_dominated_trials=include_dominated_trials,
            targets=(lambda trial: [trial.values[index] for index in targets])
            if targets is not None
            else None,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_contour(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the contour plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the color bar.
        """
        fig = optuna.visualization.plot_contour(
            mcp.study, params=params, target=lambda t: t.values[target], target_name=target_name
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_parallel_coordinate(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the parallel coordinate plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label and the legend.
        """
        fig = optuna.visualization.plot_parallel_coordinate(
            mcp.study,
            params=params,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_slice(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the slice plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label.
        """
        fig = optuna.visualization.plot_slice(
            mcp.study,
            params=params,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_param_importances(
        params: list[str] | None = None,
        target: int | None = None,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the parameter importances plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
                By default, all objective will be plotted by setting target to None.
            target_name:
                Target’s name to display on the legend.
        """
        evaluator = optuna.importance.PedAnovaImportanceEvaluator()
        fig = optuna.visualization.plot_param_importances(
            mcp.study,
            evaluator=evaluator,
            params=params,
            target=(lambda t: t.values[target]) if target is not None else None,
            target_name=target_name,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_edf(
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the EDF plot as an image.

        Args:
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label.
        """
        fig = optuna.visualization.plot_edf(
            mcp.study,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return _render_png(fig)

    @mcp.tool()
    def plot_timeline() -> Image:
        """Return the timeline plot as an image."""
        fig = optuna.visualization.plot_timeline(mcp.study)
        return _render_png(fig)

    @mcp.tool()
    def plot_rank(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the rank plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the color bar.
        """
        fig = optuna.visualization.plot_rank(
            mcp.study, params=params, target=lambda t: t.values[target], target_name=target_name
        )
        return _render_png(fig)

    @mcp.tool(structured_output=True)
    def launch_optuna_dashboard(port: int = 58080) -> str:
        """Launch the Optuna dashboard"""
        storage: str | optuna.storages.BaseStorage | None = None
        if mcp.dashboard_thread_port is not None:
            return f"Optuna dashboard is already running. Open http://127.0.0.1:{mcp.dashboard_thread_port[1]}."

        if mcp.study is not None:
            storage = mcp.study._storage
        elif mcp.storage is not None:
            storage = mcp.storage
        else:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No study has been created. Please create a study first.",
                )
            )

        def runner(storage: optuna.storages.BaseStorage | str, port: int) -> None:
            try:
                optuna_dashboard.run_server(storage=storage, host="127.0.0.1", port=port)
            except Exception as e:
                print(f"Error starting the dashboard: {e}", file=sys.stderr)
                sys.exit(1)

        # TODO(y0z): Consider better implementation
        thread = threading.Thread(
            target=runner,
            args=(storage, port),
            daemon=True,
        )
        thread.start()
        mcp.dashboard_thread_port = (thread, port)

        return f"Optuna dashboard is running at http://127.0.0.1:{port}"

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="optuna-mcp",
        description="A server for managing Optuna studies using the MCP.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="The storage URL for Optuna studies. If not specified, it will use the in-memory storage.",
    )
    args = parser.parse_args()

    mcp = OptunaMCP("Optuna", storage=args.storage)
    mcp = register_tools(mcp)
    if sys.platform == "win32":
        # Load OpenBLAS before stdio starts its blocking read on Windows.
        # Loading it for the first time inside GPSampler can stall the request.
        import_module("scipy.linalg")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
