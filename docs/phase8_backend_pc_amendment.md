# Prospective shared-backend and Python PC amendment

P8-BACKEND-PC-20261004 is authorized by the current user instruction. No P8-S scientific fit, score or nozzle ODE study result exists yet in this checkout. Commit this amendment before any new training.

Torch CPU float64 training/scoring is primary. MLX float32 training with NumPy CPU64 scoring is optional on Mac. Backend selection uses --backend or CATJET_ML_BACKEND, default torch. Dense checkpoints are framework-neutral NPZ. Torch supports optional L-BFGS polish; the primary nozzle retains its registered Adam/L-BFGS budgets, whereas optional MLX uses Adam and discloses the missing polish.

P8-S uses N64/256/1024/4096, M-data/M-phys and three fixed seeds (24 fits). Registered losses, split/label budgets, fidelity thresholds and sole-score protection remain unchanged. PC simulation uses Python v6, with a20-row frozen parity smoke at rtol1e-9 and actual1/10-worker timing. No C++ build or Mac-specific command is required on PC.

See the JSON amendment for exact parent/amended hashes and execution scope. Historical running-chain registrations/evidence in the original checkout are untouched.
