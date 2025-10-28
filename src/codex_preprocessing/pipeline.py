import logging
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from codex_preprocessing._constants import CodexFiles
from codex_preprocessing.data import CodexDataset
from codex_preprocessing.nodes import Node

log = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline for sequential CODEX preprocessing.

    Manages the execution flow between preprocessing nodes, ensuring each node
    receives the output from the previous node as input. Nodes can be skipped
    by setting them to None.

    Processing order:
        1. deconvolution
        2. edof
        3. illumination_correction
        4. stitching (optional, can be skipped)
        5. background_correction
        6. tma_dearray

    Args:
        data: Initial input dataset.
        out_dir: Base output directory for all processing steps.
        remove_intermediate: If True, removes previous node outputs after
            successful execution to save disk space. Critical outputs
            (background_correction, tma_dearray) are always preserved.
    """

    # Define the processing order
    NODE_ORDER = [
        CodexFiles.DECONVOLUTION,
        CodexFiles.EDOF,
        CodexFiles.ILLUMINATION_CORRECTION,
        CodexFiles.STITCHING,
        CodexFiles.BACKGROUND_CORRECTION,
        CodexFiles.TMA_DEARRAY,
    ]

    KEEP_NODES = {
        CodexFiles.BACKGROUND_CORRECTION,  # Keep background corrected data
        CodexFiles.TMA_DEARRAY,  # Keep TMA dearrayed data
    }

    def __init__(
        self,
        deconvolution: Optional[Callable] = None,
        edof: Optional[Callable] = None,
        illumination_correction: Optional[Callable] = None,
        stitching: Optional[Callable] = None,
        background_correction: Optional[Callable] = None,
        tma_dearray: Optional[Callable] = None,
        data: Optional[CodexDataset] = None,
        out_dir: Optional[str | Path] = None,
        remove_intermediate: bool = False,
    ):
        self.out_dir = Path(out_dir) if out_dir else None
        self.initial_data = data
        self.remove_intermediate = remove_intermediate

        # Store node factories (not instances yet)
        self.node_factories = {
            CodexFiles.DECONVOLUTION: deconvolution,
            CodexFiles.EDOF: edof,
            CodexFiles.ILLUMINATION_CORRECTION: illumination_correction,
            CodexFiles.STITCHING: stitching,
            CodexFiles.BACKGROUND_CORRECTION: background_correction,
            CodexFiles.TMA_DEARRAY: tma_dearray,
        }

        # Track pipeline state
        self.current_state = None
        self.current_data = data
        self.executed_nodes = []
        self.node_outputs = {}  # Track output directories

        self._log_pipeline_config()

    def _log_pipeline_config(self):
        """Log configured pipeline nodes for execution tracking."""
        active_nodes = [name for name in self.NODE_ORDER if self.node_factories[name] is not None]

        if not active_nodes:
            log.warning("No nodes configured in pipeline.")
            return

        log.info(f"Pipeline configured with {len(active_nodes)} node(s):")
        for i, name in enumerate(active_nodes, 1):
            log.info(f"  {i}. {name}")

        if self.remove_intermediate:
            log.info("Intermediate outputs will be removed after each step")
        else:
            log.info("All intermediate outputs will be preserved")

    def _get_node_output_dir(self, node_name: str) -> Path:
        """Get the output directory for a specific node."""
        if self.out_dir is None:
            raise ValueError("Output directory not set.")
        return self.out_dir / node_name

    def _remove_previous_output(self):
        """
        Remove the output directory of the previous node to save disk space.

        Called after a node completes successfully. Removes the output of the
        node before the current one (second-to-last in executed_nodes).
        """
        if not self.remove_intermediate:
            return

        # Need at least 2 executed nodes to have a "previous" node
        if len(self.executed_nodes) < 2:
            return

        # Get the node BEFORE the current one (second to last in the list)
        prev_node = self.executed_nodes[-2]
        if prev_node in self.KEEP_NODES:
            log.info(f"Keeping {prev_node} output (marked as critical)")
            return

        # Get the output directory of the previous node
        prev_output_dir = self.node_outputs.get(prev_node)

        if prev_output_dir and prev_output_dir.exists():
            try:
                log.info(f"Removing intermediate output: {prev_output_dir}")
                shutil.rmtree(prev_output_dir)
                log.info(f"Removed {prev_node} output")
            except Exception as e:
                log.warning(f"Failed to remove {prev_output_dir}: {e}")
        else:
            log.debug(f"No output directory to remove for {prev_node}")

    def _load_data_for_node(self, node_name: str) -> CodexDataset:
        """
        Load the appropriate dataset for a node.

        Returns the output from the previous executed node, or the initial
        data if this is the first node.
        """
        if not self.executed_nodes:
            # First node - use initial data
            log.info(f"Loading initial dataset for {node_name}")
            return self.initial_data
        else:
            # Load from previous node's output
            prev_node = self.executed_nodes[-1]
            prev_output_dir = self.node_outputs[prev_node]
            log.info(f"Loading dataset for {node_name} from {prev_node} output: {prev_output_dir}")

            # Create new dataset from previous output
            return CodexDataset(
                root_dir=prev_output_dir,
                mode="raw",
                lazy_loading=False,
                read_markers=False,
            )

    def _execute_node(self, node_name: str) -> bool:
        """
        Execute a single node in the pipeline.

        Args:
            node_name: Name of the node to execute.

        Returns:
            True if execution was successful, False if node was skipped.
        """
        node_factory = self.node_factories.get(node_name)

        if node_factory is None:
            # log.info(f"Skipping {node_name} (not configured)")
            return False

        log.info(f"{'='*60}")
        log.info(f"Starting {node_name}")
        log.info(f"{'='*60}")

        # Load appropriate data for this node
        self.current_data = self._load_data_for_node(node_name)
        log.info(f"\n{self.current_data.df.head()}")

        output_dir = self._get_node_output_dir(node_name)
        log.info(f"Node output directory: {output_dir}")

        # Update state
        self.current_state = node_name

        regions = self.current_data.get_unique_regions()
        log.info(f"Found regions: {regions}")
        tic = time.time()

        for region in regions:
            region_data = self.current_data.make_filtered_copy("region", region)
            assert len(region_data.get_unique_regions()) == 1, "region must be unique."
            log.info(f"Processing region {region}/{len(regions)}")

            # Create node instance with current data and output directory
            node: Node = node_factory(ds=region_data, out_dir=self.out_dir)

            # Execute the node
            region_tic = time.time()
            node.run()
            region_elapsed = time.time() - region_tic
            log.info(f"Region {region} completed in {region_elapsed/3600:.3f} hrs")

        # Track execution
        self.executed_nodes.append(node_name)
        self.node_outputs[node_name] = output_dir

        elapsed = time.time() - tic
        log.info(f"Completed {node_name} in {elapsed/3600:.3f} hrs")
        log.info(f"Output saved to: {output_dir}")

        # Remove previous node output if configured
        self._remove_previous_output()

        return True

    def run(self):
        """
        Execute the pipeline state machine.

        Processes all configured nodes in sequence, with each node receiving
        the output from the previous node as input. Skipped nodes (set to None)
        are automatically bypassed, and the data flows to the next active node.
        """
        log.info("Starting pipeline execution...")
        log.info(f"Base output directory: {self.out_dir}")

        start_time = time.time()

        for node_name in self.NODE_ORDER:
            self._execute_node(node_name)

        total_time = time.time() - start_time

        log.info(f"{'='*60}")
        log.info(f"Pipeline execution complete!")
        log.info(f"{'='*60}")
        log.info(f"Total time: {total_time/3600:.3f} hrs")
        log.info(f"Execution order: {' -> '.join(self.executed_nodes)}")

    def get_current_state(self) -> Optional[str]:
        """Get the current state (node being executed) of the pipeline."""
        return self.current_state

    def get_executed_nodes(self) -> list:
        """Get list of nodes that have been executed."""
        return self.executed_nodes.copy()

    def get_output_dir(self, node_name: str) -> Optional[Path]:
        """Get the output directory for a specific executed node."""
        return self.node_outputs.get(node_name)
