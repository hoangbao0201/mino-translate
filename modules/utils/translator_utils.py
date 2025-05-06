import stanza
from typing import List
from .textblock import TextBlock

def format_translations(blk_list: List[TextBlock], trg_lng_cd: str, upper_case: bool =True):
    for blk in blk_list:
        translation = blk.translation
        if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):

            if trg_lng_cd == 'zh-TW':
                trg_lng_cd = 'zh-Hant'
            elif trg_lng_cd == 'zh-CN':
                trg_lng_cd = 'zh-Hans'
            else:
                trg_lng_cd = trg_lng_cd

            stanza.download(trg_lng_cd, processors='tokenize')
            nlp = stanza.Pipeline(trg_lng_cd, processors='tokenize')
            doc = nlp(translation)
            seg_result = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    seg_result.append(word.text)
            translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
            blk.translation = translation
        else:
            if upper_case and not translation.isupper():
                blk.translation = translation.upper() 
            elif not upper_case and translation.isupper():
                blk.translation = translation.capitalize()
            else:
                blk.translation = translation