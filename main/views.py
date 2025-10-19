import json
import base64
import io
from collections import Counter

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from .models import EmotionRecord

from PIL import Image
import numpy as np
import cv2

# DeepFace импорт — ортаңда дұрыс орнатылғанын тексер
from deepface import DeepFace

# --- кеңес мәтіндерінің толық базасы (қазақша, креативті ұзын мәтін) ---
ADVICES = {
    "5–8 жас": {
        "Қуаныш": (
            "Керемет! Сенің күлкің айналадағы адамдарға жылулық сыйлайды. "
            "Қуанышты сәттерді ата-анаңмен және достарыңмен бөліс. "
            "Ойна, ән айта, сурет сал — бұл сенің ішкі қуатыңды арттырады. "
            "Егер жаңа іске кіріссең, оны қуанған көзқараспен істеген жақсы!"
        ),
        "Ашу": (
            "Ашу — бұл күштің белгісі, бірақ оны дұрыс басқарған жөн. "
            "Егер ашулансаң, бір сәт тыныштануға тырыс: терең дем алып, санада онға дейін сана. "
            "Ашуыңды айтуға қиын болса, сақта: ойынды өзгертіп, немесе сурет салған жақсы."
        ),
        "Қайғы": (
            "Қайғы — уақытша сезім. Қиындықтар болғанда ата-анаңа немесе ұстазға айт. "
            "Сен жалғыз емессің. Қатты қиналғанда сүйікті ойыныңды ойна немесе әңгімелес."
        ),
        "Қорқыныш": (
            "Қорқыныш — қауіптен сақтайтын сезім. Бірақ оны басу үшін ата-анаңа жағына бер, "
            "және бір-екі тыныс алу жаттығуын жаса. Қорқыныштың себептерін бірге талқылаңдар."
        ),
        "Бейтарап": (
            "Бейтарап күйде болу — бұл дем жасау уақыты. Бұл кезде жаңа ойын ойлап көр немесе кітап аш. "
            "Шағын қызықты тапсырма көңіліңді көтеруі мүмкін."
        ),
        "Таңқалу": (
            "Таң қалар нәрселер — бұл қызықтың бастауы. Қызық нәрсені зертте және сұрақтар қой. "
            "Сен кішкентай зерттеушісің!"
        ),
    },
    "8–12 жас": {
        "Қуаныш": (
            "Сенің қуанышың өмірге шабыт әкеледі. Бұл энергияны сабақта, шығармашылықта қолдан. "
            "Достарыңа көмектес, шағын мақсаттар қойып, соларды орындаған сайын қуан."
        ),
        "Ашу": (
            "Ашу — күшіңнің белгісі, бірақ оны айналаңа зиян келтірмей шығару керек. "
            "Спортпен айналысу, серуендеу немесе ойынды ауыстыру көмектеседі. "
            "Тыныс алу жаттығулары — нағыз көмектесуші құрал."
        ),
        "Қайғы": (
            "Сенің сезімдерің маңызды. Егер қайғыңды жеңе алмасаң, ата-анаңа немесе ұстазыңа айт. "
            "Ойланып, жазып көр; кішігірім әрекеттер ойыңды жеңілдетеді."
        ),
        "Қорқыныш": (
            "Қорқыныштан шыққан жол — оны бөлісу. Қорқыныштың дәл себебін ата-анаңмен талқыла. "
            "Сен кішкентай қадамдармен оны жеңе аласың."
        ),
        "Бейтарап": (
            "Бейтарап күй сенің көңіл-күйің тұрақты екенін көрсетеді. "
            "Бірақ аздап оригиналдық қосып көр: жаңа хобби немесе спорт түрін байқап көр."
        ),
        "Таңқалу": (
            "Таңқалу — сенің ішкі қызығушылығыңды оятады. Бұл кезде жаңа білім алудың тамаша мүмкіндігі туындайды."
        ),
    },
    "12–17 жас": {
        "Қуаныш": (
            "Позитивті энергияң — сенің ең үлкен күшің. Мақсат қойып іске кіріс, "
            "және осы қуанышты энергияны айналаңа таратып оқы. "
            "Жеке дамуға уақыт бөл, ол сенің келешегіңе үлкен қазына болады."
        ),
        "Ашу": (
            "Ашуды басқару — ересек болудың белгісі. Қысқа мерзімде шешім қабылдама. "
            "Жаттығу, жазу немесе сенің ойларыңды сенімді адаммен талқылау көмектеседі."
        ),
        "Қайғы": (
            "Қиын кезеңдер болады және олар сенің тұлғаңды қалыптастырады. "
            "Досқа, отбасыңа айт. Кішкентай қадамдар арқылы сен бұл кезеңнен шыға аласың."
        ),
        "Қорқыныш": (
            "Қорқыныш — өсу мен тәуекелдің көрсеткіші. Кішкентай қадамдар жасап, сен өз сенімділігіңді арттыра аласың. "
            "Көмек сұраудан қашпа — сен жалғыз емессің."
        ),
        "Бейтарап": (
            "Тыныштық — ойды жинақтаудың тамаша сәті. Осы уақытты жоспар құруға, өзін-өзі дамытуға арна."
        ),
        "Таңқалу": (
            "Таңқалу — шығармашылық пен зерттеудің басы. Қызығушылықты дамыт, сұрақтар қой, әлемді зертте."
        ),
    }
}

# Helper: map DeepFace names to our labels
EMOTION_MAP = {
    'happy': 'Қуаныш',
    'angry': 'Ашу',
    'sad': 'Қайғы',
    'fear': 'Қорқыныш',
    'neutral': 'Бейтарап',
    'surprise': 'Таңқалу',
    'disgust': 'Бейтарап'  # map disgust to neutral for simplicity
}

def analyzer_page(request):
    return render(request, 'main/analyzer.html')


def analyze(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('POST only')

    try:
        payload = json.loads(request.body)
    except Exception as e:
        return JsonResponse({'error': 'JSON decode error: ' + str(e)}, status=400)

    frames = payload.get('frames', [])
    full_name = payload.get('full_name', '—')
    age_group = payload.get('age_group', '')

    if not frames:
        return JsonResponse({'error': 'No frames provided'}, status=400)

    # collect per-frame dominant emotions and confidences
    detected_emotions = []
    per_frame_confidence = []  # confidence for dominant emotion per frame (0..100)

    for idx, fdata in enumerate(frames):
        try:
            header, b64 = fdata.split(',', 1)
            img_bytes = base64.b64decode(b64)
            # Read to numpy image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # DeepFace analyze
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            # analysis can return dict or list depending on version
            if isinstance(analysis, list):
                a = analysis[0]
            else:
                a = analysis

            dominant = a.get('dominant_emotion')  # e.g. 'happy'
            emotions_scores = a.get('emotion') or {}
            conf = None
            # Some versions report emotion percentages; try to extract
            if dominant and emotions_scores:
                conf_val = emotions_scores.get(dominant, None)
                if conf_val is not None:
                    conf = float(conf_val)
            # fallback: if no confidence, set default 50
            conf = conf if conf is not None else 50.0

            # Normalize emotion name to our language
            mapped = EMOTION_MAP.get(dominant, 'Бейтарап')
            detected_emotions.append(mapped)
            per_frame_confidence.append(conf)

        except Exception as e:
            # ignore frame errors but continue
            print("frame processing error:", e)
            continue

    if not detected_emotions:
        # nothing detected
        advice_text = "Камера арқылы эмоция анықталмады — тыныштандырып, камераға қараңыз да тағы көріңіз."
        # Save neutral with zero
        EmotionRecord.objects.create(full_name=full_name, age_group=age_group, emotion='Бейтарап', avg_score=0.0, advice=advice_text)
        return JsonResponse({"emotion": "Бейтарап", "avg_score": 0.0, "advice": advice_text})

    # find dominant overall
    counts = Counter(detected_emotions)
    dominant_emotion = counts.most_common(1)[0][0]

    # average confidence for frames where dominant occurred OR average of all confidences
    # simpler: compute average of confidences (0..100)
    avg_confidence = sum(per_frame_confidence) / len(per_frame_confidence)
    avg_percent = round(avg_confidence, 1)

    # Prepare advice (long text) from ADVICES by age_group and dominant_emotion
    advice_text = ADVICES.get(age_group, {}).get(dominant_emotion, "")

    if not advice_text:
        # fallback general advice
        advice_text = (
            "Сенің жағдайыңды нақты бағалау үшін қосымша талдау қажет. "
            "Егер сен қобалжысаң, ата-анаға немесе ұстазға айтыңыз. "
            "Күш пен батылдық сенікі!"
        )

    # Save to DB
    try:
        EmotionRecord.objects.create(
            full_name=full_name,
            age_group=age_group,
            emotion=dominant_emotion,
            avg_score=avg_percent,
            advice=advice_text
        )
    except Exception as e:
        print("DB save error:", e)

    return JsonResponse({
        "emotion": dominant_emotion,
        "avg_score": avg_percent,
        "advice": advice_text
    })
