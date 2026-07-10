"""Prompt templates per prompt language (string.Template, $-substitution)."""

from __future__ import annotations

from string import Template

# Keys: system, unassisted, rag, tool_call, output_buffer, web_search
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
        "web_search": Template(
            "Quote $reference from the $translation_name ($translation_id) "
            "translation of the Bible, exactly word for word. First call the "
            "search_web tool to find the exact text of the passage in this "
            "translation, then reproduce it exactly as found."
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
        "web_search": Template(
            "\u8bf7\u9010\u5b57\u5f15\u7528\u300a\u5723\u7ecf\u300b"
            "$translation_name\uff08$translation_id\uff09\u8bd1\u672c\u4e2d\u7684 "
            "$reference\u3002\u8bf7\u5148\u8c03\u7528 search_web \u5de5\u5177"
            "\u5728\u7f51\u4e0a\u641c\u7d22\u8be5\u8bd1\u672c\u4e2d\u8fd9\u6bb5"
            "\u7ecf\u6587\u7684\u51c6\u786e\u6587\u672c\uff0c\u7136\u540e\u539f"
            "\u6837\u9010\u5b57\u5f15\u7528\u3002"
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
        "output_buffer": Template(
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
