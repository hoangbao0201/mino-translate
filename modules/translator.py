import re
import json
import numpy as np
from typing import List
from openai import OpenAI
from .utils.textblock import TextBlock

client = OpenAI(
    api_key = "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",
)

def set_texts_from_json(blk_list: List[TextBlock], json_string: str):
    match = re.search(r"\{[\s\S]*\}", json_string)
    if match:
        # Extract the JSON string from the matched regular expression
        json_string = match.group(0)
        translation_dict = json.loads(json_string)
        
        for idx, blk in enumerate(blk_list):
            block_key = f"block_{idx}"
            if block_key in translation_dict:
                blk.translation = translation_dict[block_key]
            else:
                print(f"Warning: {block_key} not found in JSON string.")
    else:
        print("No JSON found in the input string.")

def get_raw_text(blk_list: List[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def translate_with_chatgpt(user_prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Lỗi dịch: {str(e)}"

def translate(source_lang, target_lang, blk_list: List[TextBlock]):
    entire_raw_text = get_raw_text(blk_list)

    if source_lang == "Chinese":
        user_prompt = f"""Dịch đoạn văn sau một cách tự nhiên, giữ đúng phong cách võ hiệp, trang nhã và đầy khí phách:\n{entire_raw_text}"""
        system_prompt = f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        
        **Yêu cầu quan trọng**:
        - **Trả về đúng JSON như đầu vào**, chỉ thay đổi nội dung phần text.
        - Không được làm mất cấu trúc JSON.
        - Không dịch khóa JSON (ví dụ: `block_0`, `block_1` phải giữ nguyên).
        - KHÔNG đưa ra bất kỳ giải thích nào, chỉ trả về JSON."""

    elif source_lang == "Korean":
        user_prompt = f"Make the translation sound as natural as possible.\nTranslate this\n{entire_raw_text}"
        system_prompt = f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        
        ### **RULES FOR TRANSLATION:**
        - **Proper nouns (names of people, places, or organizations) MUST BE ROMANIZED using the Revised Romanization of {source_lang} (RR).**
        - **Examples of correct Romanization:**
        - '송중기' → 'Song Joong-ki'
        - '김지수' → 'Kim Ji-soo'
        - '박해일' → 'Park Hae-il'
        - '이민호' → 'Lee Min-ho'
        - '정수정' → 'Jung Soo-jung'
        - **DO NOT translate or modify names. Keep them in their original Romanized form.**
        - **For Korean honorifics (님, 씨, 선생님, 회장님, etc.), translate them into the appropriate {target_lang} equivalent.**
        - '디렉터님' → 'GIÁM ĐỐC'
        - '사장님' → 'TỔNG GIÁM ĐỐC'
        - '선생님' → 'THẦY/CÔ'"""

    else:
        user_prompt = f"Make the translation sound as natural as possible.\nTranslate this\n{entire_raw_text}"
        system_prompt = f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations"""

    # Debug log
    print(f"\n--- system_prompt ---\n{system_prompt}\n")

    refined_translation = translate_with_chatgpt(user_prompt, system_prompt)

    print("user_prompt: ", user_prompt)
    print("refined_translation: ", refined_translation)

    set_texts_from_json(blk_list, refined_translation)

    return blk_list
