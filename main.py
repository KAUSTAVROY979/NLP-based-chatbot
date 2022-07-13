from tkinter import *
from chat import get_response, bot_name

bg_color = "#f5f5f5"
text = "#000000"

font = "Helvetica 14"

class chatgui:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Roy's Kitchen")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=570, height=550, bg =bg_color)

        #writing the label for the gui
        head_label = Label(self.window, bg="#1a0b7d",fg="#D51313",text="Welcome to Roy's Kitchen", font="Helvetica 13 bold",pady=10)
        #setting the width of label...if relwidth set to 1 width will be equal to maxwidth
        head_label.place(relwidth=1)
        #a divider between label and rest of the window
        line = Label(self.window, width=450, bg="#07044a")
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        #text
        """
        widget Used as a class variable as we need it for another function later
        """
        self.text_widget = Text(self.window, width=20, height=2, bg=bg_color, fg=text, font=font, padx=5, pady=5)
        self.text_widget.place(relheight=0.745,relwidth=1,rely=0.08)
        #more arguments for customiztion
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        #scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1,relx=0.974)
        #this will allow us to change the view of the y axis of the text_widget
        scrollbar.configure(command=self.text_widget.yview)
        
        #bottom label
        self.bottom_label = Label(self.window, bg = "#DB7093",height=80)
        self.bottom_label.place(relwidth=1,rely=0.825)

        #message sending feature
        self.msg = Entry(self.bottom_label,bg="#F08080",fg=text, font=font)
        self.msg.place(relwidth=0.74, relheight=0.06,rely=0.008, relx=0.011)
        #focus() automatically selects the message sending box by default
        self.msg.focus()
        self.msg.bind("<Return>",self.send)

        #send button for the chatbot
        send_btn = Button(self.bottom_label, text="Send",font="Helvetica 13 bold",width=20,bg="#DB7093",
                            command = lambda : self.send(None))

        send_btn.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def send(self,event):
        msg = self.msg.get()
        self.insert_msg(msg,"You")

    
    def insert_msg(self,msg,sender):
        #if message box is empty
        if not msg:
            return
        
        self.msg.delete(0,END)
        msg1 = f"{sender} : {msg}\n\n"
        """
        text area is disabled above therefore
        we have to momentarily enable it to display 
        the message
        """
        self.text_widget.configure(state = NORMAL)
        self.text_widget.insert(END,msg1)
        self.text_widget.configure(state = DISABLED)

        msg2 = f"{bot_name} : {get_response(msg)}\n\n"
        self.text_widget.configure(state = NORMAL)
        self.text_widget.insert(END,msg2)
        self.text_widget.configure(state = DISABLED)

        #scrolls to the last message
        self.text_widget.see(END)


app = chatgui()
app.run()