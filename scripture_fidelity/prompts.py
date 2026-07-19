"""Prompt templates per prompt language (string.Template, $-substitution)."""

from __future__ import annotations

from string import Template

# Keys: system, unassisted, rag, tool_call, buffer_transform, web_search
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
        "buffer_transform": Template(
            "I need $reference from the $translation_name ($translation_id) "
            "translation of the Bible. Do not write out the passage text "
            "yourself. Instead, output exactly this placeholder between the "
            "<quote> tags: {{QUOTE:$reference}} \u2014 it will be replaced "
            "programmatically with the passage text."
        ),
        "web_search": Template(
            "Quote $reference from the $translation_name ($translation_id) "
            "translation of the Bible, exactly word for word. First call the "
            "search_web tool to find the exact text of the passage in this "
            "translation, then reproduce it exactly as found."
        ),
        # Selection scenario: the passage is described indirectly; the model
        # must emit a structured reference of its own choosing. The expected
        # reference is deliberately absent from this prompt.
        "buffer_transform_selection": Template(
            "I need a passage from the $translation_name ($translation_id) "
            "translation of the Bible: $description. Decide which passage "
            "this describes. Do not write out the passage text yourself. "
            "Instead, output between the <quote> tags exactly one "
            "placeholder of the form {{QUOTE:<reference>}}, where "
            "<reference> is the Scripture reference you selected \u2014 it "
            "will be looked up and replaced programmatically with the "
            "passage text."
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
        "buffer_transform": Template(
            "\u6211\u9700\u8981\u300a\u5723\u7ecf\u300b$translation_name"
            "\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 $reference\u3002"
            "\u4e0d\u8981\u81ea\u5df1\u5199\u51fa\u7ecf\u6587\u5185\u5bb9\u3002"
            "\u8bf7\u5728 <quote> \u6807\u7b7e\u4e4b\u95f4\u8f93\u51fa\u4ee5\u4e0b"
            "\u5360\u4f4d\u7b26\uff1a{{QUOTE:$reference}} \u2014 \u5b83\u5c06\u88ab"
            "\u7a0b\u5e8f\u81ea\u52a8\u66ff\u6362\u4e3a\u7ecf\u6587\u6587\u672c\u3002"
        ),
        "web_search": Template(
            "\u8bf7\u9010\u5b57\u5f15\u7528\u300a\u5723\u7ecf\u300b"
            "$translation_name\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 "
            "$reference\u3002\u8bf7\u5148\u8c03\u7528 search_web \u5de5\u5177"
            "\u5728\u7f51\u4e0a\u641c\u7d22\u8be5\u8bd1\u672c\u4e2d\u8fd9\u6bb5"
            "\u7ecf\u6587\u7684\u51c6\u786e\u6587\u672c\uff0c\u7136\u540e\u539f"
            "\u6837\u9010\u5b57\u5f15\u7528\u3002"
        ),
        "buffer_transform_selection": Template(
            "\u6211\u9700\u8981\u300a\u5723\u7ecf\u300b$translation_name"
            "\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684\u4e00\u6bb5"
            "\u7ecf\u6587\uff1a$description\u3002\u8bf7\u5224\u65ad\u8fd9\u6307"
            "\u7684\u662f\u54ea\u6bb5\u7ecf\u6587\u3002\u4e0d\u8981\u81ea\u5df1"
            "\u5199\u51fa\u7ecf\u6587\u5185\u5bb9\u3002\u8bf7\u5728 <quote> "
            "\u6807\u7b7e\u4e4b\u95f4\u53ea\u8f93\u51fa\u4e00\u4e2a\u5f62\u5982 "
            "{{QUOTE:<\u7ecf\u6587\u51fa\u5904>}} \u7684\u5360\u4f4d\u7b26\uff0c"
            "\u5176\u4e2d <\u7ecf\u6587\u51fa\u5904> \u662f\u4f60\u9009\u5b9a\u7684"
            "\u5723\u7ecf\u51fa\u5904 \u2014 \u5b83\u5c06\u88ab\u67e5\u627e\u5e76"
            "\u7a0b\u5e8f\u81ea\u52a8\u66ff\u6362\u4e3a\u7ecf\u6587\u6587\u672c\u3002"
        ),
    },
    "spa": {
        "system": Template(
            "Eres un asistente preciso para citar la Biblia. Cuando se te pida "
            "citar un pasaje, escribe únicamente el texto del pasaje entre las "
            "etiquetas <quote> y </quote>. Dentro de las etiquetas no incluyas "
            "números de versículo, títulos, notas al pie ni ningún comentario."
        ),
        "unassisted": Template(
            "Cita $reference de la traducción $translation_name "
            "($translation_id) de la Biblia, exactamente palabra por palabra."
        ),
        "rag": Template(
            "Cita $reference de la traducción $translation_name "
            "($translation_id) de la Biblia, exactamente palabra por palabra."
            "\n\nAquí está el texto autorizado del pasaje:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Reprodúcelo exactamente como se te ha dado."
        ),
        "tool_call": Template(
            "Cita $reference de la traducción $translation_name "
            "($translation_id) de la Biblia, exactamente palabra por palabra. "
            "Primero llama a la herramienta get_passage para obtener el texto "
            "exacto y luego reprodúcelo exactamente como se devuelve."
        ),
        "buffer_transform": Template(
            "Necesito $reference de la traducción $translation_name "
            "($translation_id) de la Biblia. No escribas tú mismo el texto del "
            "pasaje. En su lugar, escribe exactamente este marcador entre las "
            "etiquetas <quote>: {{QUOTE:$reference}} — será reemplazado "
            "programáticamente por el texto del pasaje."
        ),
        "web_search": Template(
            "Cita $reference de la traducción $translation_name "
            "($translation_id) de la Biblia, exactamente palabra por palabra. "
            "Primero llama a la herramienta search_web para encontrar el texto "
            "exacto del pasaje en esta traducción y luego reprodúcelo "
            "exactamente como lo encontraste."
        ),
        "buffer_transform_selection": Template(
            "Necesito un pasaje de la traducción $translation_name "
            "($translation_id) de la Biblia: $description. Decide qué pasaje "
            "describe esto. No escribas tú mismo el texto del pasaje. En su "
            "lugar, escribe entre las etiquetas <quote> exactamente un "
            "marcador de la forma {{QUOTE:<referencia>}}, donde <referencia> "
            "es la referencia bíblica que seleccionaste — será buscada y "
            "reemplazada programáticamente por el texto del pasaje."
        ),
    },
    "fra": {
        "system": Template(
            "Tu es un assistant précis pour citer la Bible. Lorsqu'on te "
            "demande de citer un passage, n'écris que le texte du passage "
            "entre les balises <quote> et </quote>. À l'intérieur des balises, "
            "n'inclus pas de numéros de versets, de titres, de notes de bas de "
            "page ni aucun commentaire."
        ),
        "unassisted": Template(
            "Cite $reference de la traduction $translation_name "
            "($translation_id) de la Bible, exactement mot pour mot."
        ),
        "rag": Template(
            "Cite $reference de la traduction $translation_name "
            "($translation_id) de la Bible, exactement mot pour mot.\n\n"
            "Voici le texte officiel du passage :\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Reproduis-le exactement tel quel."
        ),
        "tool_call": Template(
            "Cite $reference de la traduction $translation_name "
            "($translation_id) de la Bible, exactement mot pour mot. Appelle "
            "d'abord l'outil get_passage pour récupérer le texte exact, puis "
            "reproduis-le exactement tel qu'il est renvoyé."
        ),
        "buffer_transform": Template(
            "J'ai besoin de $reference de la traduction $translation_name "
            "($translation_id) de la Bible. N'écris pas toi-même le texte du "
            "passage. Écris plutôt exactement ce marqueur entre les balises "
            "<quote> : {{QUOTE:$reference}} — il sera remplacé "
            "programmatiquement par le texte du passage."
        ),
        "web_search": Template(
            "Cite $reference de la traduction $translation_name "
            "($translation_id) de la Bible, exactement mot pour mot. Appelle "
            "d'abord l'outil search_web pour trouver le texte exact du passage "
            "dans cette traduction, puis reproduis-le exactement tel que "
            "trouvé."
        ),
        "buffer_transform_selection": Template(
            "J'ai besoin d'un passage de la traduction $translation_name "
            "($translation_id) de la Bible : $description. Détermine quel "
            "passage cela décrit. N'écris pas toi-même le texte du passage. "
            "Écris plutôt entre les balises <quote> exactement un marqueur de "
            "la forme {{QUOTE:<référence>}}, où <référence> est la référence "
            "biblique que tu as choisie — elle sera recherchée et remplacée "
            "programmatiquement par le texte du passage."
        ),
    },
    "deu": {
        "system": Template(
            "Du bist ein präziser Assistent zum Zitieren der Bibel. Wenn du "
            "gebeten wirst, eine Passage zu zitieren, gib nur den Passagentext "
            "zwischen den Tags <quote> und </quote> aus. Innerhalb der Tags "
            "dürfen keine Versnummern, Überschriften, Fußnoten oder Kommentare "
            "enthalten sein."
        ),
        "unassisted": Template(
            "Zitiere $reference aus der Übersetzung $translation_name "
            "($translation_id) der Bibel, exakt Wort für Wort."
        ),
        "rag": Template(
            "Zitiere $reference aus der Übersetzung $translation_name "
            "($translation_id) der Bibel, exakt Wort für Wort.\n\n"
            "Hier ist der maßgebliche Text der Passage:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Gib ihn exakt so wieder, wie er vorgegeben ist."
        ),
        "tool_call": Template(
            "Zitiere $reference aus der Übersetzung $translation_name "
            "($translation_id) der Bibel, exakt Wort für Wort. Rufe zuerst das "
            "Werkzeug get_passage auf, um den exakten Text abzurufen, und gib "
            "ihn dann exakt so wieder, wie er zurückgegeben wird."
        ),
        "buffer_transform": Template(
            "Ich benötige $reference aus der Übersetzung $translation_name "
            "($translation_id) der Bibel. Schreibe den Passagentext nicht "
            "selbst. Gib stattdessen genau diesen Platzhalter zwischen den "
            "<quote>-Tags aus: {{QUOTE:$reference}} — er wird programmatisch "
            "durch den Passagentext ersetzt."
        ),
        "web_search": Template(
            "Zitiere $reference aus der Übersetzung $translation_name "
            "($translation_id) der Bibel, exakt Wort für Wort. Rufe zuerst das "
            "Werkzeug search_web auf, um den exakten Text der Passage in "
            "dieser Übersetzung zu finden, und gib ihn dann exakt so wieder, "
            "wie du ihn gefunden hast."
        ),
        "buffer_transform_selection": Template(
            "Ich benötige eine Passage aus der Übersetzung $translation_name "
            "($translation_id) der Bibel: $description. Entscheide, welche "
            "Passage damit beschrieben wird. Schreibe den Passagentext nicht "
            "selbst. Gib stattdessen zwischen den <quote>-Tags genau einen "
            "Platzhalter der Form {{QUOTE:<Referenz>}} aus, wobei <Referenz> "
            "die von dir gewählte Bibelstelle ist — sie wird nachgeschlagen "
            "und programmatisch durch den Passagentext ersetzt."
        ),
    },
    "hin": {
        "system": Template(
            "आप बाइबल उद्धृत करने के लिए एक सटीक सहायक हैं। जब आपसे कोई अंश "
            "उद्धृत करने के लिए कहा जाए, तो केवल अंश का पाठ <quote> और "
            "</quote> टैग के बीच लिखें। टैग के अंदर पद संख्या, शीर्षक, "
            "पाद-टिप्पणियाँ या कोई टिप्पणी शामिल न करें।"
        ),
        "unassisted": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "$reference को शब्दशः, बिल्कुल ज्यों का त्यों उद्धृत करें।"
        ),
        "rag": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "$reference को शब्दशः, बिल्कुल ज्यों का त्यों उद्धृत करें।\n\n"
            "यह अंश का आधिकारिक पाठ है:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "इसे बिल्कुल वैसे ही प्रस्तुत करें जैसा दिया गया है।"
        ),
        "tool_call": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "$reference को शब्दशः, बिल्कुल ज्यों का त्यों उद्धृत करें। पहले "
            "get_passage टूल को कॉल करके सटीक पाठ प्राप्त करें, फिर उसे "
            "बिल्कुल वैसे ही प्रस्तुत करें जैसा लौटाया गया है।"
        ),
        "buffer_transform": Template(
            "मुझे बाइबल के $translation_name ($translation_id) अनुवाद से "
            "$reference चाहिए। अंश का पाठ स्वयं न लिखें। इसके बजाय <quote> "
            "टैग के बीच बिल्कुल यह प्लेसहोल्डर लिखें: {{QUOTE:$reference}} — "
            "इसे प्रोग्राम द्वारा अंश के पाठ से बदल दिया जाएगा।"
        ),
        "web_search": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "$reference को शब्दशः, बिल्कुल ज्यों का त्यों उद्धृत करें। पहले "
            "search_web टूल को कॉल करके इस अनुवाद में अंश का सटीक पाठ खोजें, "
            "फिर उसे बिल्कुल वैसे ही प्रस्तुत करें जैसा मिला है।"
        ),
        "buffer_transform_selection": Template(
            "मुझे बाइबल के $translation_name ($translation_id) अनुवाद से "
            "एक अंश चाहिए: $description। तय करें कि यह किस अंश का वर्णन "
            "करता है। अंश का पाठ स्वयं न लिखें। इसके बजाय <quote> टैग के "
            "बीच {{QUOTE:<संदर्भ>}} रूप का ठीक एक प्लेसहोल्डर लिखें, जहाँ "
            "<संदर्भ> आपके द्वारा चुना गया शास्त्र संदर्भ है — इसे खोजकर "
            "प्रोग्राम द्वारा अंश के पाठ से बदल दिया जाएगा।"
        ),
    },
    "ara": {
        "system": Template(
            "أنت مساعد دقيق لاقتباس الكتاب المقدس. عندما يُطلب منك اقتباس "
            "مقطع، اكتب نص المقطع فقط بين الوسمين <quote> و</quote>. لا "
            "تُدرج داخل الوسمين أرقام الآيات أو العناوين أو الحواشي أو أي "
            "تعليق."
        ),
        "unassisted": Template(
            "اقتبس $reference من ترجمة $translation_name ($translation_id) "
            "للكتاب المقدس، حرفيًا كلمة بكلمة."
        ),
        "rag": Template(
            "اقتبس $reference من ترجمة $translation_name ($translation_id) "
            "للكتاب المقدس، حرفيًا كلمة بكلمة.\n\n"
            "إليك النص المعتمد للمقطع:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "أعد كتابته تمامًا كما هو معطى."
        ),
        "tool_call": Template(
            "اقتبس $reference من ترجمة $translation_name ($translation_id) "
            "للكتاب المقدس، حرفيًا كلمة بكلمة. استدعِ أولاً أداة get_passage "
            "للحصول على النص الدقيق، ثم أعد كتابته تمامًا كما أُعيد."
        ),
        "buffer_transform": Template(
            "أحتاج إلى $reference من ترجمة $translation_name "
            "($translation_id) للكتاب المقدس. لا تكتب نص المقطع بنفسك. بدلاً "
            "من ذلك، اكتب هذا العنصر النائب بالضبط بين وسمي <quote>: "
            "{{QUOTE:$reference}} — سيتم استبداله برمجيًا بنص المقطع."
        ),
        "web_search": Template(
            "اقتبس $reference من ترجمة $translation_name ($translation_id) "
            "للكتاب المقدس، حرفيًا كلمة بكلمة. استدعِ أولاً أداة search_web "
            "للعثور على النص الدقيق للمقطع في هذه الترجمة، ثم أعد كتابته "
            "تمامًا كما وجدته."
        ),
        "buffer_transform_selection": Template(
            "أحتاج إلى مقطع من ترجمة $translation_name ($translation_id) "
            "للكتاب المقدس: $description. حدِّد أي مقطع يصفه هذا. لا تكتب نص "
            "المقطع بنفسك. بدلاً من ذلك، اكتب بين وسمي <quote> عنصرًا نائبًا "
            "واحدًا بالضبط بالشكل {{QUOTE:<المرجع>}}، حيث <المرجع> هو المرجع "
            "الكتابي الذي اخترته — سيتم البحث عنه واستبداله برمجيًا بنص "
            "المقطع."
        ),
    },
    "por": {
        "system": Template(
            "Você é um assistente preciso para citar a Bíblia. Quando for "
            "solicitado a citar uma passagem, escreva apenas o texto da "
            "passagem entre as tags <quote> e </quote>. Dentro das tags, não "
            "inclua números de versículos, títulos, notas de rodapé nem "
            "qualquer comentário."
        ),
        "unassisted": Template(
            "Cite $reference da tradução $translation_name ($translation_id) "
            "da Bíblia, exatamente palavra por palavra."
        ),
        "rag": Template(
            "Cite $reference da tradução $translation_name ($translation_id) "
            "da Bíblia, exatamente palavra por palavra.\n\n"
            "Aqui está o texto oficial da passagem:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Reproduza-o exatamente como fornecido."
        ),
        "tool_call": Template(
            "Cite $reference da tradução $translation_name ($translation_id) "
            "da Bíblia, exatamente palavra por palavra. Primeiro chame a "
            "ferramenta get_passage para obter o texto exato e depois "
            "reproduza-o exatamente como retornado."
        ),
        "buffer_transform": Template(
            "Preciso de $reference da tradução $translation_name "
            "($translation_id) da Bíblia. Não escreva você mesmo o texto da "
            "passagem. Em vez disso, escreva exatamente este marcador entre "
            "as tags <quote>: {{QUOTE:$reference}} — ele será substituído "
            "programaticamente pelo texto da passagem."
        ),
        "web_search": Template(
            "Cite $reference da tradução $translation_name ($translation_id) "
            "da Bíblia, exatamente palavra por palavra. Primeiro chame a "
            "ferramenta search_web para encontrar o texto exato da passagem "
            "nesta tradução e depois reproduza-o exatamente como encontrado."
        ),
        "buffer_transform_selection": Template(
            "Preciso de uma passagem da tradução $translation_name "
            "($translation_id) da Bíblia: $description. Decida qual passagem "
            "isto descreve. Não escreva você mesmo o texto da passagem. Em "
            "vez disso, escreva entre as tags <quote> exatamente um marcador "
            "da forma {{QUOTE:<referência>}}, onde <referência> é a "
            "referência bíblica que você selecionou — ela será consultada e "
            "substituída programaticamente pelo texto da passagem."
        ),
    },
    "urd": {
        "system": Template(
            "آپ بائبل کے اقتباس کے لیے ایک درست معاون ہیں۔ جب آپ سے کوئی "
            "اقتباس نقل کرنے کو کہا جائے تو صرف اقتباس کا متن <quote> اور "
            "</quote> ٹیگز کے درمیان لکھیں۔ ٹیگز کے اندر آیت کے نمبر، "
            "عنوانات، حواشی یا کوئی تبصرہ شامل نہ کریں۔"
        ),
        "unassisted": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے "
            "$reference کو لفظ بہ لفظ، بالکل درست نقل کریں۔"
        ),
        "rag": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے "
            "$reference کو لفظ بہ لفظ، بالکل درست نقل کریں۔\n\n"
            "اقتباس کا مستند متن یہ ہے:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "اسے بالکل ویسا ہی نقل کریں جیسا دیا گیا ہے۔"
        ),
        "tool_call": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے "
            "$reference کو لفظ بہ لفظ، بالکل درست نقل کریں۔ پہلے "
            "get_passage ٹول کو کال کر کے درست متن حاصل کریں، پھر اسے بالکل "
            "ویسا ہی نقل کریں جیسا واپس کیا گیا ہے۔"
        ),
        "buffer_transform": Template(
            "مجھے بائبل کے $translation_name ($translation_id) ترجمے سے "
            "$reference چاہیے۔ اقتباس کا متن خود نہ لکھیں۔ اس کے بجائے "
            "<quote> ٹیگز کے درمیان بالکل یہ پلیس ہولڈر لکھیں: "
            "{{QUOTE:$reference}} — اسے پروگرام کے ذریعے اقتباس کے متن سے "
            "بدل دیا جائے گا۔"
        ),
        "web_search": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے "
            "$reference کو لفظ بہ لفظ، بالکل درست نقل کریں۔ پہلے "
            "search_web ٹول کو کال کر کے اس ترجمے میں اقتباس کا درست متن "
            "تلاش کریں، پھر اسے بالکل ویسا ہی نقل کریں جیسا ملا ہے۔"
        ),
        "buffer_transform_selection": Template(
            "مجھے بائبل کے $translation_name ($translation_id) ترجمے سے "
            "ایک اقتباس چاہیے: $description۔ فیصلہ کریں کہ یہ کس اقتباس کا "
            "بیان ہے۔ اقتباس کا متن خود نہ لکھیں۔ اس کے بجائے <quote> ٹیگز "
            "کے درمیان {{QUOTE:<حوالہ>}} کی شکل کا بالکل ایک پلیس ہولڈر "
            "لکھیں، جہاں <حوالہ> وہ کتابی حوالہ ہے جو آپ نے منتخب کیا — اسے "
            "تلاش کر کے پروگرام کے ذریعے اقتباس کے متن سے بدل دیا جائے گا۔"
        ),
    },
    "rus": {
        "system": Template(
            "Ты — точный помощник по цитированию Библии. Когда тебя просят "
            "процитировать отрывок, выводи только текст отрывка между тегами "
            "<quote> и </quote>. Внутри тегов не включай номера стихов, "
            "заголовки, сноски или какие-либо комментарии."
        ),
        "unassisted": Template(
            "Процитируй $reference из перевода Библии $translation_name "
            "($translation_id) в точности слово в слово."
        ),
        "rag": Template(
            "Процитируй $reference из перевода Библии $translation_name "
            "($translation_id) в точности слово в слово.\n\n"
            "Вот авторитетный текст отрывка:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "Воспроизведи его в точности так, как дано."
        ),
        "tool_call": Template(
            "Процитируй $reference из перевода Библии $translation_name "
            "($translation_id) в точности слово в слово. Сначала вызови "
            "инструмент get_passage, чтобы получить точный текст, затем "
            "воспроизведи его в точности так, как он возвращён."
        ),
        "buffer_transform": Template(
            "Мне нужен $reference из перевода Библии $translation_name "
            "($translation_id). Не пиши текст отрывка сам. Вместо этого "
            "выведи между тегами <quote> ровно этот плейсхолдер: "
            "{{QUOTE:$reference}} — он будет программно заменён текстом "
            "отрывка."
        ),
        "web_search": Template(
            "Процитируй $reference из перевода Библии $translation_name "
            "($translation_id) в точности слово в слово. Сначала вызови "
            "инструмент search_web, чтобы найти точный текст отрывка в этом "
            "переводе, затем воспроизведи его в точности так, как найдено."
        ),
        "buffer_transform_selection": Template(
            "Мне нужен отрывок из перевода Библии $translation_name "
            "($translation_id): $description. Определи, какой отрывок это "
            "описывает. Не пиши текст отрывка сам. Вместо этого выведи между "
            "тегами <quote> ровно один плейсхолдер вида "
            "{{QUOTE:<ссылка>}}, где <ссылка> — выбранная тобой ссылка на "
            "Писание — она будет найдена и программно заменена текстом "
            "отрывка."
        ),
    },
    "ben": {
        "system": Template(
            "আপনি বাইবেল উদ্ধৃত করার জন্য একজন নির্ভুল সহায়ক। যখন আপনাকে "
            "কোনো অংশ উদ্ধৃত করতে বলা হয়, তখন শুধুমাত্র অংশের পাঠ্য <quote> "
            "এবং </quote> ট্যাগের মধ্যে লিখুন। ট্যাগের ভিতরে পদ সংখ্যা, "
            "শিরোনাম, পাদটীকা বা কোনো মন্তব্য অন্তর্ভুক্ত করবেন না।"
        ),
        "unassisted": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "$reference হুবহু শব্দে শব্দে উদ্ধৃত করুন।"
        ),
        "rag": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "$reference হুবহু শব্দে শব্দে উদ্ধৃত করুন।\n\n"
            "এখানে অংশটির প্রামাণিক পাঠ্য রয়েছে:\n\n"
            "<passage>\n$context\n</passage>\n\n"
            "এটি ঠিক যেমন দেওয়া হয়েছে তেমনই পুনরুত্পাদন করুন।"
        ),
        "tool_call": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "$reference হুবহু শব্দে শব্দে উদ্ধৃত করুন। প্রথমে get_passage "
            "টুল কল করে সঠিক পাঠ্য সংগ্রহ করুন, তারপর যেমন ফেরত এসেছে ঠিক "
            "তেমনই পুনরুত্পাদন করুন।"
        ),
        "buffer_transform": Template(
            "আমার বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "$reference প্রয়োজন। অংশের পাঠ্য নিজে লিখবেন না। পরিবর্তে "
            "<quote> ট্যাগের মধ্যে ঠিক এই প্লেসহোল্ডারটি লিখুন: "
            "{{QUOTE:$reference}} — এটি প্রোগ্রামের মাধ্যমে অংশের পাঠ্য "
            "দিয়ে প্রতিস্থাপিত হবে।"
        ),
        "web_search": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "$reference হুবহু শব্দে শব্দে উদ্ধৃত করুন। প্রথমে search_web "
            "টুল কল করে এই অনুবাদে অংশটির সঠিক পাঠ্য খুঁজুন, তারপর যেমন "
            "পাওয়া গেছে ঠিক তেমনই পুনরুত্পাদন করুন।"
        ),
        "buffer_transform_selection": Template(
            "আমার বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "একটি অংশ প্রয়োজন: $description। ঠিক করুন এটি কোন অংশকে বর্ণনা "
            "করে। অংশের পাঠ্য নিজে লিখবেন না। পরিবর্তে <quote> ট্যাগের মধ্যে "
            "{{QUOTE:<সূত্র>}} আকারের ঠিক একটি প্লেসহোল্ডার লিখুন, যেখানে "
            "<সূত্র> আপনার নির্বাচিত শাস্ত্রীয় সূত্র — এটি খুঁজে নিয়ে "
            "প্রোগ্রামের মাধ্যমে অংশের পাঠ্য দিয়ে প্রতিস্থাপিত হবে।"
        ),
    },
}

# Multi-reference variants: the prompt asks for several passages at once and
# requires one attributed quote block per passage. Placeholders: $references
# (joined list), $translation_name, $translation_id, $context (rag),
# $example (buffer_transform placeholder example).
MULTI_PROMPTS: dict[str, dict[str, Template]] = {
    "eng": {
        "system": Template(
            "You are a precise assistant for quoting the Bible. When asked "
            "to quote passages, output each passage in its own quote block "
            'of the form <quote ref="REFERENCE">passage text</quote>, using '
            "exactly the reference you were given. Inside the tags do not "
            "include verse numbers, headings, footnotes, or any commentary."
        ),
        "unassisted": Template(
            "Quote each of the following passages from the $translation_name "
            "($translation_id) translation of the Bible, exactly word for "
            "word: $references."
        ),
        "rag": Template(
            "Quote each of the following passages from the $translation_name "
            "($translation_id) translation of the Bible, exactly word for "
            "word: $references.\n\n"
            "Here are the authoritative texts of the passages:\n\n"
            "$context\n\n"
            "Reproduce each exactly as given."
        ),
        "tool_call": Template(
            "Quote each of the following passages from the $translation_name "
            "($translation_id) translation of the Bible, exactly word for "
            "word: $references. First call the get_passage tool once for "
            "each reference to retrieve its exact text, then reproduce each "
            "exactly as returned."
        ),
        "buffer_transform": Template(
            "I need each of the following passages from the "
            "$translation_name ($translation_id) translation of the Bible: "
            "$references. Do not write out any passage text yourself. "
            "Instead, output for each passage exactly one placeholder in its "
            "own quote block, like this: $example \u2014 each placeholder "
            "will be replaced programmatically with the passage text."
        ),
        "web_search": Template(
            "Quote each of the following passages from the $translation_name "
            "($translation_id) translation of the Bible, exactly word for "
            "word: $references. First call the search_web tool to find the "
            "exact text of each passage in this translation, then reproduce "
            "each exactly as found."
        ),
    },
    "zho": {
        "system": Template(
            "你是一位精确引用圣经的助手。当被要求引用多段经文时，请将每段"
            "经文分别输出在各自的引用块中，格式为 <quote ref=\"经文出处\">"
            "经文正文</quote>，ref 属性必须与给定的出处完全一致。标签内"
            "不要包含节号、标题、脚注或任何评论。"
        ),
        "unassisted": Template(
            "请逐字引用《圣经》$translation_name（$translation_id）译本中的"
            "以下各段经文：$references。"
        ),
        "rag": Template(
            "请逐字引用《圣经》$translation_name（$translation_id）译本中的"
            "以下各段经文：$references。\n\n以下是这些经文的权威文本：\n\n"
            "$context\n\n请完全按照给出的文本原样引用每一段。"
        ),
        "tool_call": Template(
            "请逐字引用《圣经》$translation_name（$translation_id）译本中的"
            "以下各段经文：$references。请先为每个出处分别调用一次 "
            "get_passage 工具获取准确经文，然后原样逐字引用每一段。"
        ),
        "buffer_transform": Template(
            "我需要《圣经》$translation_name（$translation_id）译本中的以下"
            "各段经文：$references。不要自己写出经文内容。请为每段经文在"
            "各自的引用块中输出一个占位符，例如：$example — 每个占位符"
            "将被程序自动替换为经文文本。"
        ),
        "web_search": Template(
            "请逐字引用《圣经》$translation_name（$translation_id）译本中的"
            "以下各段经文：$references。请先调用 search_web 工具在网上搜索"
            "该译本中每段经文的准确文本，然后原样逐字引用每一段。"
        ),
    },
    "spa": {
        "system": Template(
            "Eres un asistente preciso para citar la Biblia. Cuando se te "
            "pida citar varios pasajes, escribe cada pasaje en su propio "
            'bloque de cita con la forma <quote ref="REFERENCIA">texto del '
            "pasaje</quote>, usando exactamente la referencia que se te dio. "
            "Dentro de las etiquetas no incluyas números de versículo, "
            "títulos, notas al pie ni ningún comentario."
        ),
        "unassisted": Template(
            "Cita cada uno de los siguientes pasajes de la traducción "
            "$translation_name ($translation_id) de la Biblia, exactamente "
            "palabra por palabra: $references."
        ),
        "rag": Template(
            "Cita cada uno de los siguientes pasajes de la traducción "
            "$translation_name ($translation_id) de la Biblia, exactamente "
            "palabra por palabra: $references.\n\n"
            "Aquí están los textos autorizados de los pasajes:\n\n"
            "$context\n\n"
            "Reproduce cada uno exactamente como se te ha dado."
        ),
        "tool_call": Template(
            "Cita cada uno de los siguientes pasajes de la traducción "
            "$translation_name ($translation_id) de la Biblia, exactamente "
            "palabra por palabra: $references. Primero llama a la "
            "herramienta get_passage una vez por cada referencia para "
            "obtener su texto exacto y luego reproduce cada uno exactamente "
            "como se devuelve."
        ),
        "buffer_transform": Template(
            "Necesito cada uno de los siguientes pasajes de la traducción "
            "$translation_name ($translation_id) de la Biblia: $references. "
            "No escribas tú mismo el texto de ningún pasaje. En su lugar, "
            "escribe para cada pasaje exactamente un marcador en su propio "
            "bloque de cita, así: $example — cada marcador será reemplazado "
            "programáticamente por el texto del pasaje."
        ),
        "web_search": Template(
            "Cita cada uno de los siguientes pasajes de la traducción "
            "$translation_name ($translation_id) de la Biblia, exactamente "
            "palabra por palabra: $references. Primero llama a la "
            "herramienta search_web para encontrar el texto exacto de cada "
            "pasaje en esta traducción y luego reproduce cada uno "
            "exactamente como lo encontraste."
        ),
    },
    "fra": {
        "system": Template(
            "Tu es un assistant précis pour citer la Bible. Lorsqu'on te "
            "demande de citer plusieurs passages, écris chaque passage dans "
            'son propre bloc de citation de la forme <quote ref="RÉFÉRENCE">'
            "texte du passage</quote>, en utilisant exactement la référence "
            "donnée. À l'intérieur des balises, n'inclus pas de numéros de "
            "versets, de titres, de notes de bas de page ni aucun "
            "commentaire."
        ),
        "unassisted": Template(
            "Cite chacun des passages suivants de la traduction "
            "$translation_name ($translation_id) de la Bible, exactement mot "
            "pour mot : $references."
        ),
        "rag": Template(
            "Cite chacun des passages suivants de la traduction "
            "$translation_name ($translation_id) de la Bible, exactement mot "
            "pour mot : $references.\n\n"
            "Voici les textes officiels des passages :\n\n"
            "$context\n\n"
            "Reproduis chacun exactement tel quel."
        ),
        "tool_call": Template(
            "Cite chacun des passages suivants de la traduction "
            "$translation_name ($translation_id) de la Bible, exactement mot "
            "pour mot : $references. Appelle d'abord l'outil get_passage une "
            "fois pour chaque référence afin de récupérer son texte exact, "
            "puis reproduis chacun exactement tel qu'il est renvoyé."
        ),
        "buffer_transform": Template(
            "J'ai besoin de chacun des passages suivants de la traduction "
            "$translation_name ($translation_id) de la Bible : $references. "
            "N'écris toi-même le texte d'aucun passage. Écris plutôt pour "
            "chaque passage exactement un marqueur dans son propre bloc de "
            "citation, comme ceci : $example — chaque marqueur sera remplacé "
            "programmatiquement par le texte du passage."
        ),
        "web_search": Template(
            "Cite chacun des passages suivants de la traduction "
            "$translation_name ($translation_id) de la Bible, exactement mot "
            "pour mot : $references. Appelle d'abord l'outil search_web pour "
            "trouver le texte exact de chaque passage dans cette traduction, "
            "puis reproduis chacun exactement tel que trouvé."
        ),
    },
    "deu": {
        "system": Template(
            "Du bist ein präziser Assistent zum Zitieren der Bibel. Wenn du "
            "gebeten wirst, mehrere Passagen zu zitieren, gib jede Passage "
            "in einem eigenen Zitatblock der Form "
            '<quote ref="REFERENZ">Passagentext</quote> aus und verwende '
            "dabei exakt die vorgegebene Referenz. Innerhalb der Tags dürfen "
            "keine Versnummern, Überschriften, Fußnoten oder Kommentare "
            "enthalten sein."
        ),
        "unassisted": Template(
            "Zitiere jede der folgenden Passagen aus der Übersetzung "
            "$translation_name ($translation_id) der Bibel, exakt Wort für "
            "Wort: $references."
        ),
        "rag": Template(
            "Zitiere jede der folgenden Passagen aus der Übersetzung "
            "$translation_name ($translation_id) der Bibel, exakt Wort für "
            "Wort: $references.\n\n"
            "Hier sind die maßgeblichen Texte der Passagen:\n\n"
            "$context\n\n"
            "Gib jede exakt so wieder, wie sie vorgegeben ist."
        ),
        "tool_call": Template(
            "Zitiere jede der folgenden Passagen aus der Übersetzung "
            "$translation_name ($translation_id) der Bibel, exakt Wort für "
            "Wort: $references. Rufe zuerst für jede Referenz einmal das "
            "Werkzeug get_passage auf, um ihren exakten Text abzurufen, und "
            "gib dann jede exakt so wieder, wie sie zurückgegeben wird."
        ),
        "buffer_transform": Template(
            "Ich benötige jede der folgenden Passagen aus der Übersetzung "
            "$translation_name ($translation_id) der Bibel: $references. "
            "Schreibe keinen Passagentext selbst. Gib stattdessen für jede "
            "Passage genau einen Platzhalter in einem eigenen Zitatblock "
            "aus, etwa so: $example — jeder Platzhalter wird programmatisch "
            "durch den Passagentext ersetzt."
        ),
        "web_search": Template(
            "Zitiere jede der folgenden Passagen aus der Übersetzung "
            "$translation_name ($translation_id) der Bibel, exakt Wort für "
            "Wort: $references. Rufe zuerst das Werkzeug search_web auf, um "
            "den exakten Text jeder Passage in dieser Übersetzung zu finden, "
            "und gib dann jede exakt so wieder, wie du sie gefunden hast."
        ),
    },
    "hin": {
        "system": Template(
            "आप बाइबल उद्धृत करने के लिए एक सटीक सहायक हैं। जब आपसे कई अंश "
            "उद्धृत करने के लिए कहा जाए, तो प्रत्येक अंश को उसके अपने उद्धरण "
            'ब्लॉक में <quote ref="संदर्भ">अंश का पाठ</quote> के रूप में '
            "लिखें, और ref में ठीक वही संदर्भ रखें जो आपको दिया गया है। टैग "
            "के अंदर पद संख्या, शीर्षक, पाद-टिप्पणियाँ या कोई टिप्पणी शामिल "
            "न करें।"
        ),
        "unassisted": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "निम्नलिखित प्रत्येक अंश को शब्दशः, बिल्कुल ज्यों का त्यों "
            "उद्धृत करें: $references।"
        ),
        "rag": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "निम्नलिखित प्रत्येक अंश को शब्दशः, बिल्कुल ज्यों का त्यों "
            "उद्धृत करें: $references।\n\n"
            "ये अंशों के आधिकारिक पाठ हैं:\n\n"
            "$context\n\n"
            "प्रत्येक को बिल्कुल वैसे ही प्रस्तुत करें जैसा दिया गया है।"
        ),
        "tool_call": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "निम्नलिखित प्रत्येक अंश को शब्दशः, बिल्कुल ज्यों का त्यों "
            "उद्धृत करें: $references। पहले प्रत्येक संदर्भ के लिए एक बार "
            "get_passage टूल को कॉल करके उसका सटीक पाठ प्राप्त करें, फिर "
            "प्रत्येक को बिल्कुल वैसे ही प्रस्तुत करें जैसा लौटाया गया है।"
        ),
        "buffer_transform": Template(
            "मुझे बाइबल के $translation_name ($translation_id) अनुवाद से "
            "निम्नलिखित प्रत्येक अंश चाहिए: $references। किसी भी अंश का पाठ "
            "स्वयं न लिखें। इसके बजाय प्रत्येक अंश के लिए उसके अपने उद्धरण "
            "ब्लॉक में बिल्कुल एक प्लेसहोल्डर लिखें, इस तरह: $example — "
            "प्रत्येक प्लेसहोल्डर को प्रोग्राम द्वारा अंश के पाठ से बदल "
            "दिया जाएगा।"
        ),
        "web_search": Template(
            "बाइबल के $translation_name ($translation_id) अनुवाद से "
            "निम्नलिखित प्रत्येक अंश को शब्दशः, बिल्कुल ज्यों का त्यों "
            "उद्धृत करें: $references। पहले search_web टूल को कॉल करके इस "
            "अनुवाद में प्रत्येक अंश का सटीक पाठ खोजें, फिर प्रत्येक को "
            "बिल्कुल वैसे ही प्रस्तुत करें जैसा मिला है।"
        ),
    },
    "ara": {
        "system": Template(
            "أنت مساعد دقيق لاقتباس الكتاب المقدس. عندما يُطلب منك اقتباس "
            "عدة مقاطع، اكتب كل مقطع في كتلة اقتباس خاصة به بالشكل "
            '<quote ref="المرجع">نص المقطع</quote>، مستخدمًا المرجع المعطى '
            "لك بالضبط. لا تُدرج داخل الوسمين أرقام الآيات أو العناوين أو "
            "الحواشي أو أي تعليق."
        ),
        "unassisted": Template(
            "اقتبس كل واحد من المقاطع التالية من ترجمة $translation_name "
            "($translation_id) للكتاب المقدس، حرفيًا كلمة بكلمة: "
            "$references."
        ),
        "rag": Template(
            "اقتبس كل واحد من المقاطع التالية من ترجمة $translation_name "
            "($translation_id) للكتاب المقدس، حرفيًا كلمة بكلمة: "
            "$references.\n\n"
            "إليك النصوص المعتمدة للمقاطع:\n\n"
            "$context\n\n"
            "أعد كتابة كل واحد تمامًا كما هو معطى."
        ),
        "tool_call": Template(
            "اقتبس كل واحد من المقاطع التالية من ترجمة $translation_name "
            "($translation_id) للكتاب المقدس، حرفيًا كلمة بكلمة: "
            "$references. استدعِ أولاً أداة get_passage مرة واحدة لكل مرجع "
            "للحصول على نصه الدقيق، ثم أعد كتابة كل واحد تمامًا كما أُعيد."
        ),
        "buffer_transform": Template(
            "أحتاج إلى كل واحد من المقاطع التالية من ترجمة "
            "$translation_name ($translation_id) للكتاب المقدس: "
            "$references. لا تكتب نص أي مقطع بنفسك. بدلاً من ذلك، اكتب لكل "
            "مقطع عنصرًا نائبًا واحدًا بالضبط في كتلة اقتباس خاصة به، هكذا: "
            "$example — سيتم استبدال كل عنصر نائب برمجيًا بنص المقطع."
        ),
        "web_search": Template(
            "اقتبس كل واحد من المقاطع التالية من ترجمة $translation_name "
            "($translation_id) للكتاب المقدس، حرفيًا كلمة بكلمة: "
            "$references. استدعِ أولاً أداة search_web للعثور على النص "
            "الدقيق لكل مقطع في هذه الترجمة، ثم أعد كتابة كل واحد تمامًا "
            "كما وجدته."
        ),
    },
    "por": {
        "system": Template(
            "Você é um assistente preciso para citar a Bíblia. Quando for "
            "solicitado a citar várias passagens, escreva cada passagem em "
            "seu próprio bloco de citação na forma "
            '<quote ref="REFERÊNCIA">texto da passagem</quote>, usando '
            "exatamente a referência que lhe foi dada. Dentro das tags, não "
            "inclua números de versículos, títulos, notas de rodapé nem "
            "qualquer comentário."
        ),
        "unassisted": Template(
            "Cite cada uma das seguintes passagens da tradução "
            "$translation_name ($translation_id) da Bíblia, exatamente "
            "palavra por palavra: $references."
        ),
        "rag": Template(
            "Cite cada uma das seguintes passagens da tradução "
            "$translation_name ($translation_id) da Bíblia, exatamente "
            "palavra por palavra: $references.\n\n"
            "Aqui estão os textos oficiais das passagens:\n\n"
            "$context\n\n"
            "Reproduza cada uma exatamente como fornecida."
        ),
        "tool_call": Template(
            "Cite cada uma das seguintes passagens da tradução "
            "$translation_name ($translation_id) da Bíblia, exatamente "
            "palavra por palavra: $references. Primeiro chame a ferramenta "
            "get_passage uma vez para cada referência para obter seu texto "
            "exato e depois reproduza cada uma exatamente como retornada."
        ),
        "buffer_transform": Template(
            "Preciso de cada uma das seguintes passagens da tradução "
            "$translation_name ($translation_id) da Bíblia: $references. "
            "Não escreva você mesmo o texto de nenhuma passagem. Em vez "
            "disso, escreva para cada passagem exatamente um marcador em "
            "seu próprio bloco de citação, assim: $example — cada marcador "
            "será substituído programaticamente pelo texto da passagem."
        ),
        "web_search": Template(
            "Cite cada uma das seguintes passagens da tradução "
            "$translation_name ($translation_id) da Bíblia, exatamente "
            "palavra por palavra: $references. Primeiro chame a ferramenta "
            "search_web para encontrar o texto exato de cada passagem nesta "
            "tradução e depois reproduza cada uma exatamente como "
            "encontrada."
        ),
    },
    "urd": {
        "system": Template(
            "آپ بائبل کے اقتباس کے لیے ایک درست معاون ہیں۔ جب آپ سے کئی "
            "اقتباسات نقل کرنے کو کہا جائے تو ہر اقتباس کو اس کے اپنے "
            'اقتباسی بلاک میں <quote ref="حوالہ">اقتباس کا متن</quote> کی '
            "شکل میں لکھیں، اور ref میں بالکل وہی حوالہ رکھیں جو آپ کو دیا "
            "گیا ہے۔ ٹیگز کے اندر آیت کے نمبر، عنوانات، حواشی یا کوئی "
            "تبصرہ شامل نہ کریں۔"
        ),
        "unassisted": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے درج ذیل "
            "میں سے ہر اقتباس کو لفظ بہ لفظ، بالکل درست نقل کریں: "
            "$references۔"
        ),
        "rag": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے درج ذیل "
            "میں سے ہر اقتباس کو لفظ بہ لفظ، بالکل درست نقل کریں: "
            "$references۔\n\n"
            "اقتباسات کے مستند متون یہ ہیں:\n\n"
            "$context\n\n"
            "ہر ایک کو بالکل ویسا ہی نقل کریں جیسا دیا گیا ہے۔"
        ),
        "tool_call": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے درج ذیل "
            "میں سے ہر اقتباس کو لفظ بہ لفظ، بالکل درست نقل کریں: "
            "$references۔ پہلے ہر حوالے کے لیے ایک بار get_passage ٹول کو "
            "کال کر کے اس کا درست متن حاصل کریں، پھر ہر ایک کو بالکل ویسا "
            "ہی نقل کریں جیسا واپس کیا گیا ہے۔"
        ),
        "buffer_transform": Template(
            "مجھے بائبل کے $translation_name ($translation_id) ترجمے سے "
            "درج ذیل میں سے ہر اقتباس چاہیے: $references۔ کسی بھی اقتباس کا "
            "متن خود نہ لکھیں۔ اس کے بجائے ہر اقتباس کے لیے اس کے اپنے "
            "اقتباسی بلاک میں بالکل ایک پلیس ہولڈر لکھیں، اس طرح: $example "
            "— ہر پلیس ہولڈر کو پروگرام کے ذریعے اقتباس کے متن سے بدل دیا "
            "جائے گا۔"
        ),
        "web_search": Template(
            "بائبل کے $translation_name ($translation_id) ترجمے سے درج ذیل "
            "میں سے ہر اقتباس کو لفظ بہ لفظ، بالکل درست نقل کریں: "
            "$references۔ پہلے search_web ٹول کو کال کر کے اس ترجمے میں ہر "
            "اقتباس کا درست متن تلاش کریں، پھر ہر ایک کو بالکل ویسا ہی نقل "
            "کریں جیسا ملا ہے۔"
        ),
    },
    "rus": {
        "system": Template(
            "Ты — точный помощник по цитированию Библии. Когда тебя просят "
            "процитировать несколько отрывков, выводи каждый отрывок в его "
            "собственном блоке цитаты вида "
            '<quote ref="ССЫЛКА">текст отрывка</quote>, используя в точности '
            "данную тебе ссылку. Внутри тегов не включай номера стихов, "
            "заголовки, сноски или какие-либо комментарии."
        ),
        "unassisted": Template(
            "Процитируй каждый из следующих отрывков из перевода Библии "
            "$translation_name ($translation_id) в точности слово в слово: "
            "$references."
        ),
        "rag": Template(
            "Процитируй каждый из следующих отрывков из перевода Библии "
            "$translation_name ($translation_id) в точности слово в слово: "
            "$references.\n\n"
            "Вот авторитетные тексты отрывков:\n\n"
            "$context\n\n"
            "Воспроизведи каждый в точности так, как дано."
        ),
        "tool_call": Template(
            "Процитируй каждый из следующих отрывков из перевода Библии "
            "$translation_name ($translation_id) в точности слово в слово: "
            "$references. Сначала вызови инструмент get_passage по одному "
            "разу для каждой ссылки, чтобы получить её точный текст, затем "
            "воспроизведи каждый в точности так, как он возвращён."
        ),
        "buffer_transform": Template(
            "Мне нужен каждый из следующих отрывков из перевода Библии "
            "$translation_name ($translation_id): $references. Не пиши "
            "текст отрывков сам. Вместо этого выведи для каждого отрывка "
            "ровно один плейсхолдер в его собственном блоке цитаты, вот "
            "так: $example — каждый плейсхолдер будет программно заменён "
            "текстом отрывка."
        ),
        "web_search": Template(
            "Процитируй каждый из следующих отрывков из перевода Библии "
            "$translation_name ($translation_id) в точности слово в слово: "
            "$references. Сначала вызови инструмент search_web, чтобы найти "
            "точный текст каждого отрывка в этом переводе, затем "
            "воспроизведи каждый в точности так, как найдено."
        ),
    },
    "ben": {
        "system": Template(
            "আপনি বাইবেল উদ্ধৃত করার জন্য একজন নির্ভুল সহায়ক। যখন আপনাকে "
            "একাধিক অংশ উদ্ধৃত করতে বলা হয়, তখন প্রতিটি অংশ তার নিজস্ব "
            'উদ্ধৃতি ব্লকে <quote ref="সূত্র">অংশের পাঠ্য</quote> আকারে '
            "লিখুন, এবং ref-এ ঠিক সেই সূত্রটিই ব্যবহার করুন যা আপনাকে দেওয়া "
            "হয়েছে। ট্যাগের ভিতরে পদ সংখ্যা, শিরোনাম, পাদটীকা বা কোনো "
            "মন্তব্য অন্তর্ভুক্ত করবেন না।"
        ),
        "unassisted": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "নিম্নলিখিত প্রতিটি অংশ হুবহু শব্দে শব্দে উদ্ধৃত করুন: "
            "$references।"
        ),
        "rag": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "নিম্নলিখিত প্রতিটি অংশ হুবহু শব্দে শব্দে উদ্ধৃত করুন: "
            "$references।\n\n"
            "এখানে অংশগুলির প্রামাণিক পাঠ্য রয়েছে:\n\n"
            "$context\n\n"
            "প্রতিটি ঠিক যেমন দেওয়া হয়েছে তেমনই পুনরুত্পাদন করুন।"
        ),
        "tool_call": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "নিম্নলিখিত প্রতিটি অংশ হুবহু শব্দে শব্দে উদ্ধৃত করুন: "
            "$references। প্রথমে প্রতিটি সূত্রের জন্য একবার করে get_passage "
            "টুল কল করে তার সঠিক পাঠ্য সংগ্রহ করুন, তারপর প্রতিটি যেমন "
            "ফেরত এসেছে ঠিক তেমনই পুনরুত্পাদন করুন।"
        ),
        "buffer_transform": Template(
            "আমার বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "নিম্নলিখিত প্রতিটি অংশ প্রয়োজন: $references। কোনো অংশের "
            "পাঠ্য নিজে লিখবেন না। পরিবর্তে প্রতিটি অংশের জন্য তার নিজস্ব "
            "উদ্ধৃতি ব্লকে ঠিক একটি প্লেসহোল্ডার লিখুন, এইভাবে: $example — "
            "প্রতিটি প্লেসহোল্ডার প্রোগ্রামের মাধ্যমে অংশের পাঠ্য দিয়ে "
            "প্রতিস্থাপিত হবে।"
        ),
        "web_search": Template(
            "বাইবেলের $translation_name ($translation_id) অনুবাদ থেকে "
            "নিম্নলিখিত প্রতিটি অংশ হুবহু শব্দে শব্দে উদ্ধৃত করুন: "
            "$references। প্রথমে search_web টুল কল করে এই অনুবাদে প্রতিটি "
            "অংশের সঠিক পাঠ্য খুঁজুন, তারপর প্রতিটি যেমন পাওয়া গেছে ঠিক "
            "তেমনই পুনরুত্পাদন করুন।"
        ),
    },
}


def _templates(language: str, multi: bool) -> dict[str, Template]:
    source = MULTI_PROMPTS if multi else PROMPTS
    templates = source.get(language)
    if templates is None:
        kind = "multi-reference " if multi else ""
        raise ValueError(
            f"No {kind}prompt templates for language {language!r} "
            f"(available: {sorted(source)})"
        )
    return templates


def build_prompt(
    language: str,
    method: str,
    reference: str,
    translation_name: str,
    translation_id: str,
    context: str = "",
    description: str = "",
) -> str:
    return _templates(language, multi=False)[method].substitute(
        reference=reference,
        translation_name=translation_name,
        translation_id=translation_id,
        context=context,
        description=description,
    )


def build_multi_prompt(
    language: str,
    method: str,
    references: list[str],
    translation_name: str,
    translation_id: str,
    contexts: list[tuple[str, str]] | None = None,
) -> str:
    """Build a multi-reference prompt.

    ``contexts`` is a list of (reference, passage text) pairs used only by
    the rag method. The buffer_transform example shows the expected attributed
    placeholder block for the first requested reference.
    """
    context = "\n\n".join(
        f'<passage ref="{ref}">\n{text}\n</passage>' for ref, text in contexts or []
    )
    example = f'<quote ref="{references[0]}">{{{{QUOTE:{references[0]}}}}}</quote>'
    return _templates(language, multi=True)[method].substitute(
        references="; ".join(references),
        translation_name=translation_name,
        translation_id=translation_id,
        context=context,
        example=example,
    )


METHOD_SYSTEM_INSTRUCTIONS = {
    "unassisted": (
        "Do not use tools or external sources. Produce the requested quotation "
        "from model knowledge."
    ),
    "rag": (
        "Treat text inside <authoritative_source> as the source of record. "
        "Reproduce exactly the span requested inside <user_request>."
    ),
    "tool_call": (
        "You must call get_passage for the requested reference before answering, "
        "then reproduce exactly the text returned by that tool."
    ),
    "web_search": (
        "You must call search_web before answering and use the located requested "
        "translation as the quotation source."
    ),
    "buffer_transform": (
        "Do not produce passage text. Your entire response must be exactly "
        "<quote>{{QUOTE:<reference>}}</quote>, replacing <reference> with the "
        "explicitly requested reference. Preserve both opening braces and both "
        "closing braces in the placeholder."
    ),
    "buffer_transform_selection": (
        "Infer the passage reference from the user request and do not produce "
        "passage text. Your entire response must be exactly "
        "<quote>{{QUOTE:<reference>}}</quote>, replacing <reference> with your "
        "selection. Preserve both opening braces and both closing braces in the "
        "placeholder."
    ),
}


def system_prompt(
    language: str, method: str | None = None, multi: bool = False
) -> str:
    base = _templates(language, multi)["system"].substitute()
    # Generated non-English prompts already contain localized method
    # instructions. Caller-supplied prompts are currently restricted to English.
    if method is None or language != "eng":
        return base
    instruction = METHOD_SYSTEM_INSTRUCTIONS.get(method)
    if instruction is None:
        raise ValueError(f"No system instruction for method: {method}")
    return f"{base}\n\nExperimental condition: {instruction}"
