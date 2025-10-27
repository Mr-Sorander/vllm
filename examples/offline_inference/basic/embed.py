# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="intfloat/e5-small",
        # runner="pooling",         # vllm在0.4.0版本以后废除该参数，使用task参数。task=“embedding”表示嵌入，默认task=“generate”
        task="embedding"
        enforce_eager=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass runner="pooling" for embedding models
    llm = LLM(**vars(args))

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = llm.embed(prompts) # 两种写法，返回的对象不同，详见下面的代码
    outputs_v2 = llm.encode(prompts)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding  # 【调用 llm.embed(prompts)时，返回的是 EmbeddingOutput，取嵌入向量需要用 .embedding，返回的是list类型
                                           #  eg: EmbeddingRequestOutput(request_id='0', outputs=EmbeddingOutput(hidden_size=1024), prompt_token_ids=[9707, 11, 847, 829, 374, 151643], finished=True)】
        
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)

    print(f"="*80)
    print("\nGenerated Outputs_v2:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs_v2):
        embeds = output.outputs.data    # 【调用 llm.encode(prompts)时，返回的是 PoolingOutput，取嵌入向量需要用 .data，返回的是tensor类型
                                        #   eg: PoolingRequestOutput(request_id='4', outputs=PoolingOutput(data=tensor([ 0.0209, -0.0456, -0.0100,  ..., -0.0190, -0.0290,  0.0064])), prompt_token_ids=[9707, 11, 847, 829, 374, 151643], finished=True)】
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
