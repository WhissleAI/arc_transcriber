# Test sentences for the context-memory eval (as used in Steer's libarc run; all names are fictional).
APPS = ["WhatsApp", "Slack", "Messages", "Telegram", "Mail", "Safari", "Notion", "Zoom", "Discord", "Signal"]
PEOPLE = ["Priyanka Raman", "Rohan Mehta", "Anaya Desai", "Siddharth Iyer", "Mei Lin", "Tomasz Kowalski", "Oluwaseun Adeyemi"]
WORDS = ["Whissle", "Steer", "Arc Mini", "PromptingNemo", "Kubernetes", "Vercel", "Gutenberg"]
BOOK = ["Elizabeth Bennet", "Mr Darcy", "Pemberley", "Wickham", "Queequeg", "Ishmael", "Ahab"]
COMMANDS = ["Pause.", "Keep reading.", "Say that again.", "Read faster.", "How long is left?", "Sleep in twenty minutes.",
 "What time is it?", "Summarise this.", "Read what I copied.", "Mute WhatsApp for an hour.", "Mute Telegram.",
 "Open Slack.", "Mute Discord for thirty minutes.", "Open Notion.", "Reply to Priyanka saying I'm on my way.",
 "Next paragraph.", "Go back.", "Music off.", "Play rain.", "Stop listening."]
DICT = ["Send the draft to Priyanka Raman before Thursday.", "Rohan Mehta will review the Steer release tonight.",
 "Anaya Desai asked whether Arc Mini runs on the phone.", "Siddharth Iyer moved the Kubernetes cluster to Vercel.",
 "Mei Lin wants the PromptingNemo branch merged today.", "Tomasz Kowalski is presenting the Whissle demo on Monday.",
 "Oluwaseun Adeyemi found three bugs in the Gutenberg importer.", "Ask Priyanka and Mei Lin to join the Zoom call.",
 "Whissle ships Steer with Arc Mini built in.", "The Gutenberg shelf now opens books at chapter one."]
BOOKD = ["Elizabeth Bennet refused Mr Darcy at first.", "Wickham lied to Elizabeth about Mr Darcy.",
 "They finally visited Pemberley in the summer.", "Queequeg shared his bed with Ishmael at the inn.",
 "Captain Ahab chased the white whale for years."]
PAIRS = [("I met Oluwaseun Adeyemi at the conference.", "Oluwaseun showed me his new speech model."),
 ("Tomasz Kowalski joined the team last week.", "Tomasz already fixed the Kubernetes deploy."),
 ("Queequeg was a harpooner from a distant island.", "Queequeg never forgot his island."),
 ("We are testing PromptingNemo on the new cluster.", "PromptingNemo trains the director model."),
 ("Siddharth Iyer called about the Vercel bill.", "Siddharth wants to move it to a cheaper plan."),
 ("Mr Darcy finally proposed again.", "This time Elizabeth Bennet said yes.")]
# Speech said to someone else in the room (not to the assistant): the negatives for device-directed.
OTHER = ["Can you pass me the salt, please?", "Did you remember to call your mother?",
 "Honestly that movie was better than the book.", "I can't believe how cold it is today.",
 "I think we should get pizza tonight.", "I told Steer to read the news earlier.",
 "Let's take the dog for a walk after dinner.", "My brother is visiting next weekend.",
 "She said the report is due on Friday.", "Steering the boat was harder than I thought.",
 "The meeting ran long so I missed the train.", "We should leave before the traffic gets bad."]
