from transitions import Machine

class LoanApplication:
    states = ['draft', 'initial_review', 'final_review', 'approved', 'rejected']

    def __init__(self):
        self.machine = Machine(
            model=self,
            states=LoanApplication.states,
            initial='draft'
        )
        # 定义状态转移规则
        self.machine.add_transition('submit', 'draft', 'initial_review')
        self.machine.add_transition('pass_initial', 'initial_review', 'final_review')
        self.machine.add_transition('pass_final', 'final_review', 'approved')
        self.machine.add_transition('reject', ['initial_review', 'final_review'], 'rejected')

# 测试
app = LoanApplication()
app.submit()
print(app.state)  # initial_review
app.pass_initial()
print(app.state)  # final_review
app.pass_final()
print(app.state)  # approved