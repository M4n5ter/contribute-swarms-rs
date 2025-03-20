# swarms-rs

**The Enterprise-Grade, Production-Ready Multi-Agent Orchestration Framework in Rust**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

## Overview

`Swarms-rs` is an enterprise-grade, production-ready multi-agent orchestration framework built in Rust, designed to handle the most demanding tasks with unparalleled speed and efficiency. Leveraging Rust’s bleeding-edge performance and safety features, `swarms-rs` provides a powerful and scalable solution for orchestrating complex multi-agent systems across various industries.

## Key Benefits

### ⚡ **Extreme Performance**
- **Multi-Threaded Architecture**: Utilize the full potential of modern multi-core processors with Rust’s zero-cost abstractions and fearless concurrency. `Swarms-rs` ensures that your agents run with minimal overhead, achieving maximum throughput and efficiency.
- **Bleeding-Edge Speed**: Written in Rust, `swarms-rs` delivers near-zero latency and lightning-fast execution, making it the ideal choice for high-frequency and real-time applications.

### 🛡 **Enterprise-Grade Reliability**
- **Memory Safety**: Rust’s ownership model guarantees memory safety without the need for a garbage collector, ensuring that your multi-agent systems are free from data races and memory leaks.
- **Production-Ready**: Designed for real-world deployment, `swarms-rs` is ready to handle mission-critical tasks with robustness and reliability that you can depend on.

### 🧠 **Powerful Orchestration**
- **Advanced Agent Coordination**: Seamlessly manage and coordinate thousands of agents, allowing them to communicate and collaborate efficiently to achieve complex goals.
- **Extensible and Modular**: `Swarms-rs` is highly modular, allowing developers to easily extend and customize the framework to suit specific use cases.

### 🚀 **Scalable and Efficient**
- **Optimized for Scale**: Whether you’re orchestrating a handful of agents or scaling up to millions, `swarms-rs` is designed to grow with your needs, maintaining top-tier performance at every level.
- **Resource Efficiency**: Maximize the use of system resources with Rust’s fine-grained control over memory and processing power, ensuring that your agents run optimally even under heavy loads.

## Getting Started

To get started with `swarms-rs`, follow the installation and usage instructions below:

### Installation

Add `swarms-rs` to your `Cargo.toml`:

```toml
[dependencies]
swarms-rs = "1.0"
```

### Usage

Here’s a basic example to get you started:

```rust
use std::env;

use anyhow::Result;
use swarms_rs::{llm::provider::openai::OpenAI, structs::agent::Agent};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(
            tracing_subscriber::fmt::layer()
                .with_line_number(true)
                .with_file(true),
        )
        .init();

    let base_url = env::var("DEEPSEEK_BASE_URL").unwrap();
    let api_key = env::var("DEEPSEEK_API_KEY").unwrap();
    let client = OpenAI::from_url(base_url, api_key).set_model("deepseek-chat");
    let agent = client
        .agent_builder()
        .system_prompt("You are a helpful assistant.")
        .agent_name("SwarmsAgent")
        .user_name("User")
        .enable_autosave()
        .max_loops(1)
        .save_sate_path("./temp/agent1_state.json") // or "./temp", we will ignore the base file.
        .enable_plan("Split the task into subtasks.".to_owned())
        .build();
    let response = agent
        .run("What is the meaning of life?".to_owned())
        .await
        .unwrap();
    println!("{response}");

    Ok(())
}

```

For more detailed examples and advanced usage, please refer to our [documentation](link_to_docs).

## Contributing

We welcome contributions from the community! Please see our [CONTRIBUTING.md](link_to_contributing.md) for guidelines on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or feedback, please open an issue or contact us at [kye@swarms.world](mailto:kye@swarms.world).
