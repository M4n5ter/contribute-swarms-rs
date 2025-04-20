use std::env;
use std::sync::Arc;

use anyhow::Result;
use swarms_rs::llm::provider::openai::OpenAI;
use swarms_rs::structs::hierarchical_swarm::HierarchicalSwarm;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_line_number(true)
        .with_file(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let base_url = env::var("DEEPSEEK_BASE_URL").unwrap();
    let api_key = env::var("DEEPSEEK_API_KEY").unwrap();
    let client = OpenAI::from_url(base_url, api_key).set_model("deepseek-chat");

    let user = "Swarms User";

    let text_processor = client
        .agent_builder()
        .agent_name("Text Processor")
        .system_prompt(
            "You are a Text Processor Agent. Your primary function is to receive raw text and perform basic processing tasks as instructed. This might include cleaning the text (removing extra whitespace, fixing minor typos if obvious), extracting specific types of information (like names, dates, or keywords mentioned in the task), or reformatting the text. Respond with the processed text or extracted information in a clear, structured format. Do not perform analysis or summarization unless specifically asked to extract certain parts.",
        )
        .user_name(user)
        .disable_task_evaluator_tool()
        .max_loops(1)
        .temperature(0.3)
        .enable_autosave()
        .save_state_dir("./temp")
        .build();

    let data_analyst = client
        .agent_builder()
        .agent_name("Data Analyst")
        .system_prompt(
            "You are a Data Analyst Agent. Your primary function is to receive processed data or text, perform analysis, identify patterns, themes, frequencies, or sentiments based on the task instructions. You should focus on extracting meaningful insights from the data provided. Respond clearly with your findings, often in a structured format like bullet points or summaries. Do not perform tasks outside the scope of data analysis (e.g., text generation, report writing)."
        )
        .user_name(user)
        .disable_task_evaluator_tool()
        .max_loops(1)
        .temperature(0.3)
        .enable_autosave()
        .save_state_dir("./temp")
        .build();

    let report_writer = client
        .agent_builder()
        .agent_name("Report Writer")
        .system_prompt(
            "You are a Report Writer Agent. Your responsibility is to synthesize information provided to you (often from other agents like Text Processors or Data Analysts) into a coherent and well-structured report. Follow the specific instructions regarding the report's format, tone, and key points to include. Ensure the final report is clear, concise, and directly addresses the requirements outlined in the task. Do not generate new data or perform analysis yourself; rely solely on the input provided."
        )
        .user_name(user)
        .disable_task_evaluator_tool()
        .max_loops(1)
        .temperature(0.3)
        .enable_autosave()
        .save_state_dir("./temp")
        .build();

    let agents = vec![text_processor, data_analyst, report_writer]
        .into_iter()
        .map(|a| Arc::new(a) as _)
        .collect::<Vec<_>>();

    let workflow = HierarchicalSwarm::builder()
        .name("Hierarchical Swarm")
        .description("An example for Hierarchical Swarm")
        .director_agent_builder(client.agent_builder())
        .agents(agents)
        .build();

    let result = workflow.run(
    r#"**Overall Goal:** Analyze the provided text about renewable energy and generate a brief report summarizing its key points and sentiment.

    **Provided Text:**
    "Renewable energy sources like solar and wind power are becoming increasingly vital in the global effort to combat climate change. Solar panel installations have seen exponential growth due to falling costs and rising efficiency. Wind energy, particularly offshore wind farms, offers tremendous potential for generating large amounts of clean electricity. However, challenges remain, including intermittency issues (solar only works when the sun shines, wind when it blows) and the need for significant upgrades to grid infrastructure to handle these variable sources. Battery storage technology is rapidly improving, offering a potential solution to the intermittency problem. Investment in renewables continues to surge, signaling a major shift in the energy landscape, though the transition requires careful planning and sustained commitment."

    **Specific Requirements:**
    1. Process the provided text: Extract the main renewable energy sources mentioned.
    2. Analyze the processed text: Identify the main benefits and challenges discussed regarding renewable energy. Determine the overall sentiment (positive, negative, neutral/mixed) towards the renewable energy transition expressed in the text.
    3. Synthesize the findings: Write a short report (2 paragraphs) summarizing the key energy sources, their benefits, the challenges mentioned, and the overall sentiment of the provided text.

    **Rules:**
    - Base all analysis and the final report strictly on the provided text.
    - Do not introduce external information or opinions.
    - The final report should be objective and concise."#).await?;

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
