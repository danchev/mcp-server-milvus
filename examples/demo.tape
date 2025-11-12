# Source code for the VHS demo example.
#
# To run:
#
#     vhs < demo.tape

Output demo.gif
Output demo.mp4
Output demo.webm
Output frames/
Output demo.gif

Set CursorBlink false
Set PlaybackSpeed 1.0
Set TypingSpeed 75ms
Set FontSize 18
Set Width 1300
Set Height 650

Type@150ms "crush run 'What databases are available in Milvus?'"

Sleep 500ms

Enter

Sleep 6s

Type@150ms "crush run 'What collections are available in the `default` database?'"

Sleep 500ms

Enter

Sleep 6s
