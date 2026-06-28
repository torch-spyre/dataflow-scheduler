# dataflow-scheduler
DataflowScheduler is a compiler infrastructure for transforming KTIR programs into optimized, pipelined execution schedules for dataflow hardware accelerators.
The scheduler takes a KTIR program—a single module containing computation expressed in the Kernel Tiled Data Parallel (KTDP) abstraction—and transforms it into a Kernel Tiled Data Flow (KTDF) representation with explicit pipeline stages, memory hierarchy management, and high-level optimizations based on hardware-specific configurations.
