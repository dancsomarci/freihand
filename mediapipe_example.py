import warnings

import mediapipe as mp

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated",
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
)
result = hands.process(...)
if result.multi_hand_landmarks is not None:
    for hand_landmarks in result.multi_hand_landmarks:
        _ = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
else:
    print("No hand landmarks detected.")
