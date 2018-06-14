# Source from google...

import win32com.client as win
speak = win.Dispatch("SAPI.SpVoice")
a="Its wonderful, we just made a text to voice converter using 4 line of code."
speak.Speak(a)
