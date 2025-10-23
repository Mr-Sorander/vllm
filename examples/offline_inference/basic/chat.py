# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
basic.py只是展示了llm生成的基本用法，实际上，chat类的应用是带有对话上下文的。
"""

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def create_parser():
    """
    将命令行参数与脚本参数分离，使得在命令行中的就可以灵活控制采样参数。
    我估计是生产环境没有ide，所以只能用命令行启动脚本。这种环境下，如果频繁改脚本，就显得很笨拙
    """
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    """
    - parser.add_argument_group("Sampling parameters") 会在命令行帮助（help）中创建一个带标题的参数组（ArgumentGroup），用于把相关的参数按组组织起来，便于阅读和分类。
    - 它返回一个可以调用 add_argument(...) 的组对象（_ArgumentGroup），组内的参数仍由同一个 ArgumentParser 解析——也就是说它只影响文档/显示，不改变解析或校验行为。
    """
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add example params
    parser.add_argument("--chat-template-path", type=str)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    """
    从命令行的参数解析器中获取的参数，有一些是构造LLM不需要的（采样参数）
    """
    max_tokens = args.pop("max_tokens") # 【注意】这里的键值和之前添加的参数不一样,【问题是】：为什么确定这个是key？
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")

    # Create an LLM
    llm = LLM(**args)

    # Create sampling params object
    # 获取并修改采样超参，如果命令行中带了 max_tokens/temperature/top_p/top_k（非 None），就覆盖 sampling_params 中对应字段。
    # 类比basic.py中，采样参数是通过 sampling_params = SamplingParams(temperature=0.8, top_p=0.95) 设置
    sampling_params = llm.get_default_sampling_params() 
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text # 打印第一组tokens（与采样参数重的n有关，默认为1）
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs)

    # You can run batch inference with llm.chat API
    conversations = [conversation for _ in range(10)] # 虚构了10段对话，模拟并发情况

    # We turn on tqdm progress bar to verify it's indeed running batch inference
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs)

    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            chat_template = f.read()

        outputs = llm.chat(
            conversations,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
        )
        print_outputs(outputs)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
