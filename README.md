## Quick Start

```sh
uv pip install -e .
pytest
python example.py
python bench.py
```

## Todos for Speculative Decoding

Main Classes Involved:

1. `SpeculativeConfig` - Configuration management
2. `NgramProposer`, `MedusaProposer`, `EagleProposer` - Different speculative methods
3. `SpecDecodeMetadata` - Metadata tracking for speculative execution
4. Modified model runners - GPU/TPU model runners with speculative support

Key Architectural Changes:

- Extra scheduler slots for speculative tokens
- Multi-step execution (draft + verification)
- Specialized proposer classes for different speculative methods
- Enhanced attention handling for tree-based methods
- Updated sampling logic for rejection sampling

Files That Would Need Modification:

1. Configuration system
2. Scheduler component
3. Model runner execution loop
4. Input/output batch management
5. Sampling logic
6. Attention metadata handling

## vllm v1 arch

`LLM.generate()` and `send request`

```sh
LLM.generate()
  -> LLM._validate_and_add_requests()
  -> LLM.llm_engine.add_request()
  -> LLM.llm_engine.engine_core.add_request() # in multiprocing sync mode , engine_core is 'SyncMPClient'
  -> LLM.llm_engine.engine_core<SyncMPClient>._send_input() # send request to ZMQ
    -> SyncMPClient.input_socket.send_multipart() # using 'input_socket' to send request
```

`LLM.__init__` and start thread to `receive request` in True loop

```sh
LLM.__init__()
  -> LLM.llm_engine = LLMEngine.from_engine_args()
    -> LLM.llm_engine.engine_core =EngineCoreClient.make_client() # 'SyncMPClient' when multiprocing sync
    -> MPClient.__init__() 'MPClient' Base of 'SyncMPClient'
      -> launch_core_engines() # EngineCoreClient manage Engines
        -> CoreEngineProcManager.__init__()
          -> proc.start() for proc in CoreEngineProcManager.processes # proc.fn is EngineCoreProc.run_engine_core
            -> run_engine_core()
              -> EngineCoreProc.__init__()
                -> super().__init__() # is EngineCore.__init__()
                  -> EngineCore.module_executor = executor_class() # init module_executor
                    -> UniProcExecutor.driver_worker.init_worker() # driver_worker is 'WorkerWrapperBase'
                      -> WorkerWrapperBase.init_worker()
                        -> WorkerWrapperBase.worker = worker_class() # init worker, class is 'gpu_worker.Worker'
                  -> EngineCore.scheduler = Scheduler() # init scheduler
                -> input_thread.start() # thread target 'process_input_sockets'
                -> process_input_sockets()
                  -> input_socket.recv_multipart() # using 'input_socket' receive request
                  -> EngineCoreProc.input_queue.put() # then put request into queue
              -> EngineCoreProc.run_busy_loop()
                # in True loop
                -> EngineCoreProc._process_input_queue() # poll the input queue
                  -> EngineCore.add_request() # 'EngineCore' is Base of 'EngineCoreProc'
                  -> EngineCore.scheduler.add_request() # add request to 'scheduler'
                -> EngineCoreProc._process_engine_step()
                  -> EngineCore.step()
                    -> EngineCore.scheduler.schedule()
                    -> EngineCore.model_executor.execute_model() # 'UniProcExecutor' in this case
```

UniProcExecutor

```sh
UniProcExecutor.execute_model()
  -> UniProcExecutor.collective_rpc()
    -> run_method(UniProcExecutor.driver_worker, 'execute_model')
      -> WorkerWrapperBase.worker.execute_model()
        -> gpu_worker.Worker.execute_model()
          -> gpu_worker.Worker.GPUModelRunner.execute_model()
```

gpu_worker.Worker -> gpu_worker.Worker.GPUModelRunner do the heavy lift

```sh
gpu_worker.Worker.GPUModelRunner()

```
