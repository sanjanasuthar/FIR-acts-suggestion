from tkinter import Tk, Label, Entry, Text, Button, END
root=Tk()
root.title("IPC Section Suggestion")
complaint_label=Label(root, text="Enter crime description")
complaint_label.pack()
complaint_entry = Entry(root, width=100)
complaint_entry.pack()
suggest_button=Button(root,text="Get Suggestion")
suggest_button.pack()
output_text = Text(root,width=100,height=20)
output_text.pack()

root.mainloop()
