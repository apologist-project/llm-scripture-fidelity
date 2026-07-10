"""Prompt templates per prompt language (string.Template, $-substitution)."""

from __future__ import annotations

from string import Template

# Keys: system, unassisted, rag, tool_call, output_buffer
PROMPTS: dict[str, dict[str, Template]] = {
    "eng": {
        "system": Template(
            "You are a precise assistant for quoting the Bible. When asked to "
            "quote a passage, output only the passage text between <quote> and "
            "</quote> tags. Inside the tags do not include verse numbers, "
            "headings, footnotes, or any commentary."
        ),
        "unassisted": Template(
            "Quote $reference from the $translation_name ($translation_id) "
            "translation of the Bible, exactly word for word."
        ),
        "rag": Template(
            "Quote $reference from the $translation_name ($translation_id) "
            "translation of the Bible, exactly word for word.\n\n"
            "Here is the authoritative text of the passage:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Reproduce it exactly as given."
        ),
        "tool_call": Template(
            "Quote $reference from the $translation_name ($translation_id) "
            "translation of the Bible, exactly word for word. First call the "
            "get_passage tool to retrieve the exact text, then reproduce it "
            "exactly as returned."
        ),
        "output_buffer": Template(
            "I need $reference from the $translation_name ($translation_id) "
            "translation of the Bible. Do not write out the passage text "
            "yourself. Instead, output exactly this placeholder between the "
            "<quote> tags: {{QUOTE:$reference}} \u2014 it will be replaced "
            "programmatically with the passage text."
        ),
    },
    "zho": {
        "system": Template(
            "\u4f60\u662f\u4e00\u4f4d\u7cbe\u786e\u5f15\u7528\u5723\u7ecf\u7684"
            "\u52a9\u624b\u3002\u5f53\u88ab\u8981\u6c42\u5f15\u7528\u7ecf\u6587"
            "\u65f6\uff0c\u53ea\u5728 <quote> \u548c </quote> \u6807\u7b7e\u4e4b"
            "\u95f4\u8f93\u51fa\u7ecf\u6587\u6b63\u6587\u3002\u6807\u7b7e\u5185"
            "\u4e0d\u8981\u5305\u542b\u8282\u53f7\u3001\u6807\u9898\u3001\u811a"
            "\u6ce8\u6216\u4efb\u4f55\u8bc4\u8bba\u3002"
        ),
        "unassisted": Template(
            "\u8bf7\u9010\u5b57\u5f15\u7528\u300a\u5723\u7ecf\u300b"
            "$translation_name\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 "
            "$reference\u3002"
        ),
        "rag": Template(
            "\u8bf7\u9010\u5b57\u5f15\u7528\u300a\u5723\u7ecf\u300b"
            "$translation_name\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 "
            "$reference\u3002\n\n\u4ee5\u4e0b\u662f\u8be5\u6bb5\u7ecf\u6587\u7684"
            "\u6743\u5a01\u6587\u672c\uff1a\n\n<passage>\n$context\n</passage>\n\n"
            "\u8bf7\u5b8c\u5168\u6309\u7167\u7ed9\u51fa\u7684\u6587\u672c\u539f"
            "\u6837\u5f15\u7528\u3002"
        ),
        "tool_call": Template(
            "\u8bf7\u9010\u5b57\u5f15\u7528\u300a\u5723\u7ecf\u300b"
            "$translation_name\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 "
            "$reference\u3002\u8bf7\u5148\u8c03\u7528 get_passage \u5de5\u5177"
            "\u83b7\u53d6\u51c6\u786e\u7ecf\u6587\uff0c\u7136\u540e\u539f\u6837"
            "\u9010\u5b57\u5f15\u7528\u3002"
        ),
        "output_buffer": Template(
            "\u6211\u9700\u8981\u300a\u5723\u7ecf\u300b$translation_name"
            "\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 $reference\u3002"
            "\u4e0d\u8981\u81ea\u5df1\u5199\u51fa\u7ecf\u6587\u5185\u5bb9\u3002"
            "\u8bf7\u5728 <quote> \u6807\u7b7e\u4e4b\u95f4\u8f93\u51fa\u4ee5\u4e0b"
            "\u5360\u4f4d\u7b26\uff1a{{QUOTE:$reference}} \u2014 \u5b83\u5c06\u88ab"
            "\u7a0b\u5e8f\u81ea\u52a8\u66ff\u6362\u4e3a\u7ecf\u6587\u6587\u672c\u3002"
        ),
    },
}


def build_prompt(
    language: str,
    method: str,
    reference: str,
    translation_name: str,
    translation_id: str,
    context: str = "",
) -> str:
    templates = PROMPTS.get(language)
    if templates is None:
        raise ValueError(
            f"No prompt templates for language {language!r} "
            f"(available: {sorted(PROMPTS)})"
        )
    return templates[method].substitute(
        reference=reference,
        translation_name=translation_name,
        translation_id=translation_id,
        context=context,
    )


def system_prompt(language: str) -> str:
    templates = PROMPTS.get(language)
    if templates is None:
        raise ValueError(
            f"No prompt templates for language {language!r} "
            f"(available: {sorted(PROMPTS)})"
        )
    return templates["system"].substitute()
