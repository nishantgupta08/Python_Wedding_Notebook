# Fix string quoting and write the new Streamlit file again.
import streamlit as st
from textwrap import dedent
import ast
import traceback
import pandas as pd
import os
from datetime import datetime

# ============================
# Page config
# ============================
st.set_page_config(
    page_title="Python Test",
    layout="wide"
)

# ============================
# Paths (CSV persistence)
# ============================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(DATA_DIR, "submissions_summary.csv")   # one row per student
DETAILS_CSV = os.path.join(DATA_DIR, "submissions_details.csv")   # one row per task
DRAFTS_DIR = os.path.join(DATA_DIR, "drafts")                     # per-student drafts
os.makedirs(DRAFTS_DIR, exist_ok=True)

# Live feedback storage
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback.csv")

# ============================
# Submit-once check (by roll no)
# ============================

def has_submitted(roll_no: str) -> bool:
    if not os.path.exists(SUMMARY_CSV):
        return False
    try:
        df = pd.read_csv(SUMMARY_CSV, usecols=["roll_no"])
        return str(roll_no) in set(df["roll_no"].astype(str))
    except Exception:
        return False

# ============================
# CSV save helpers
# ============================

def save_submission_csv(summary_row: dict, detail_rows: list):
    df_sum = pd.DataFrame([summary_row])
    if os.path.exists(SUMMARY_CSV):
        df_sum.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df_sum.to_csv(SUMMARY_CSV, index=False)

    if detail_rows:
        df_det = pd.DataFrame(detail_rows)
        if os.path.exists(DETAILS_CSV):
            df_det.to_csv(DETAILS_CSV, mode="a", header=False, index=False)
        else:
            df_det.to_csv(DETAILS_CSV, index=False)

# ============================
# Safe execution sandbox
# ============================
ALLOWED_BUILTINS = {
    "len": len, "sorted": sorted, "range": range,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "min": min, "max": max, "sum": sum, "any": any, "all": all,
    "enumerate": enumerate,
}

def run_code(init_code: str, user_code: str):
    """Execute code with restricted builtins and return (namespace, error)."""
    ns = {"__builtins__": {}}
    ns["__builtins__"].update(ALLOWED_BUILTINS)
    try:
        exec(dedent(init_code), ns, ns)
        if user_code and user_code.strip():
            tree = ast.parse(user_code)
            banned_nodes = (
                ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Raise, ast.Global, ast.Nonlocal, ast.Lambda
            )
            for node in ast.walk(tree):
                if isinstance(node, banned_nodes):
                    raise ValueError("This type of statement isn't allowed in this exercise.")
            exec(compile(tree, filename="<user>", mode="exec"), ns, ns)
        return ns, None
    except Exception:
        return None, traceback.format_exc()

# ============================
# Task bank
# ============================
CATEGORIES = [
    {
        "key": "list",
        "title": "Lists",
        "init_code": '''\
songs_list = ["Here Comes the Bride", "Canon in D", "A Thousand Years", "Marry You"]
''',
        "show_vars": ["songs_list"],
        "tasks": [
            {"name": "Checking Existence", "desc": "Set **is_exist** to whether 'Perfect' is in songs_list.", "hint": "Use `in`.", "solution": 'is_exist = "Perfect" in songs_list', "check_var": "is_exist"},
            {"name": "Finding the Item", "desc": "Store index of 'A Thousand Years' in **index_pos**.", "hint": "`.index(value)`.", "solution": 'index_pos = songs_list.index("A Thousand Years")', "check_var": "index_pos"},
            {"name": "Adding Songs", "desc": "Append 'Perfect' to songs_list.", "hint": "`.append(...)`.", "solution": "songs_list.append('Perfect')", "check_var": "songs_list"},
            {"name": "Slicing", "desc": "Create **every_other_song** = songs_list[0:5:2]", "hint": "slice with step.", "solution": "every_other_song = songs_list[0:5:2]", "check_var": "every_other_song"},
            {"name": "Sorting", "desc": "Sort songs_list alphabetically in-place.", "hint": "`.sort()`.", "solution": "songs_list.sort()", "check_var": "songs_list"},
            {"name": "Length", "desc": "Store len(songs_list) in **num_songs**.", "hint": "`len(...)`.", "solution": "num_songs = len(songs_list)", "check_var": "num_songs"},
            {"name": "Concatenation", "desc": "Create **new_playlist** = songs_list + ['Shallow','All of Me'].", "hint": "`+` makes a new list.", "solution": "new_playlist = songs_list + ['Shallow', 'All of Me']", "check_var": "new_playlist"},
        ],
    },
    {
        "key": "dict",
        "title": "Dictionaries",
        "init_code": '''\
vendor_details = {"flowers": "1240-3343345", "wedding_rings": "4787-5636"}
''',
        "show_vars": ["vendor_details"],
        "tasks": [
            {"name": "Checking for a Key", "desc": "Set **is_food** if 'food' key exists.", "hint": "`in dict` or `.keys()`.", "solution": 'is_food = "food" in vendor_details.keys()', "check_var": "is_food"},
            {"name": "Checking for a Value", "desc": "Set **is_phone_number** if '123-56578' appears.", "hint": "`.values()`.", "solution": 'is_phone_number = "123-56578" in vendor_details.values()', "check_var": "is_phone_number"},
            {"name": "Add New Key-Value Pair", "desc": "Add 'food': '123-456-7890'.", "hint": "assignment.", "solution": 'vendor_details["food"] = "123-456-7890"', "check_var": "vendor_details"},
            {"name": "Modifying Elements", "desc": "Change 'wedding_rings' → '4787-4636'.", "hint": "assignment.", "solution": 'vendor_details["wedding_rings"] = "4787-4636"', "check_var": "vendor_details"},
            {"name": "Length", "desc": "Store count of vendors in **vendor_count**.", "hint": "`len(...)`.", "solution": 'vendor_count = len(vendor_details)', "check_var": "vendor_count"},
            {"name": "Extending (update)", "desc": "Update with extra_vendor_details = {'Hair Stylist':'123-368548' }.", "hint": "`.update(...)`.", "solution": "extra_vendor_details = {'Hair Stylist': '123-368548'}\nvendor_details.update(extra_vendor_details)", "check_var": "vendor_details"},
            {"name": "Concatenation (new dict)", "desc": "Create **combined_vendor_details** = {**vendor_details, **extra_vendor_details}.", "hint": "dict unpacking.", "solution": "extra_vendor_details = {'Hair Stylist': '123-368548'}\ncombined_vendor_details = {**vendor_details, **extra_vendor_details}", "check_var": "combined_vendor_details"},
        ],
    },
    {
        "key": "tuple",
        "title": "Tuples",
        "init_code": '''\
bridesmaid_role = ("Sarika", "Ritika")
''',
        "show_vars": ["bridesmaid_role"],
        "tasks": [
            {"name": "Finding the Index", "desc": "Index of 'Sarika' → **bridesmaid_index**.", "hint": "`.index(...)`.", "solution": 'bridesmaid_index = bridesmaid_role.index("Sarika")', "check_var": "bridesmaid_index"},
            {"name": "Checking Existence", "desc": "Is 'Ritika' present? → **is_bridesmaid**.", "hint": "`in`.", "solution": 'is_bridesmaid = "Ritika" in bridesmaid_role', "check_var": "is_bridesmaid"},
            {"name": "Concatenation", "desc": "**new_bridesmaid_role** = bridesmaid_role + ('Monika',).", "hint": "remember trailing comma.", "solution": "new_bridesmaid_role = bridesmaid_role + ('Monika',)", "check_var": "new_bridesmaid_role"},
        ],
    },
    {
        "key": "set",
        "title": "Sets",
        "init_code": '''\
guest_set = {"Sarika", "Divya", "Kavita"}
''',
        "show_vars": ["guest_set"],
        "tasks": [
            {"name": "Finding the Guest", "desc": "'Divya' in guest_set? → **is_present**.", "hint": "`in`.", "solution": 'is_present = "Divya" in guest_set', "check_var": "is_present"},
            {"name": "Adding the Guests", "desc": "Add 'Sarita' to guest_set.", "hint": "`.add(...)`.", "solution": 'guest_set.add("Sarita")', "check_var": "guest_set"},
            {"name": "Length", "desc": "**guest_count** = len(guest_set).", "hint": "`len(...)`.", "solution": 'guest_count = len(guest_set)', "check_var": "guest_count"},
            {"name": "Extending (update)", "desc": "Update guest_set with guest_set_father = {'Ram','Shyam' }.", "hint": "`.update(set)`.", "solution": "guest_set_father = {'Ram','Shyam'}\nguest_set.update(guest_set_father)", "check_var": "guest_set"},
            {"name": "Concatenation (new set)", "desc": "**combined_guest_set** = guest_set ∪ guest_set_father.", "hint": "`.union(...)`.", "solution": "guest_set_father = {'Ram','Shyam'}\ncombined_guest_set = guest_set.union(guest_set_father)", "check_var": "combined_guest_set"},
        ],
    },
    {
        "key": "string",
        "title": "Strings",
        "init_code": '''\
wedding_hashtag_1 = " #TheNotebookLoveStory "
wedding_hashtag_2 = "#Wedlock"
invitation_location = "Banqet Hall"
task_string = "decorations, wedding_hashtag, dresses, makeup_artist"
elite_guest_name = "Mr. Goyal"
''',
        "show_vars": ["wedding_hashtag_1", "wedding_hashtag_2", "invitation_location", "task_string", "elite_guest_name"],
        "tasks": [
            {"name": "String Stripping", "desc": "Trim spaces around wedding_hashtag_1.", "hint": "`.strip()`.", "solution": 'wedding_hashtag_1 = wedding_hashtag_1.strip()', "check_var": "wedding_hashtag_1"},
            {"name": "Accessing via Index", "desc": "`first_letter = wedding_hashtag_1[0]`.", "hint": "indexing.", "solution": 'first_letter = wedding_hashtag_1[0]', "check_var": "first_letter"},
            {"name": "Checking Existence", "desc": "'Love' in wedding_hashtag_1? → **is_love**.", "hint": "`in`.", "solution": 'is_love = "Love" in wedding_hashtag_1', "check_var": "is_love"},
            {"name": "Sorting", "desc": "**sorted_hashtag** = sorted(wedding_hashtag_1).", "hint": "`sorted(str)`.", "solution": 'sorted_hashtag = sorted(wedding_hashtag_1)', "check_var": "sorted_hashtag"},
            {"name": "Concatenation", "desc": "**invitation_location_and_date** from invitation_location + ',Saturday, August 12th'.", "hint": "string `+`.", "solution": 'invitation_location_and_date = invitation_location + "," + "Saturday, August 12th"', "check_var": "invitation_location_and_date"},
            {"name": "Replacing", "desc": "Fix 'Banqet' → 'Banquet' in invitation_location_and_date.", "hint": "`.replace()`.", "solution": 'invitation_location_and_date = invitation_location_and_date.replace("Banqet", "Banquet")', "check_var": "invitation_location_and_date"},
            {"name": "Splitting", "desc": "**task_list** = task_string.split(', ').", "hint": "`.split(', ')`.", "solution": "task_list = task_string.split(', ')", "check_var": "task_list"},
            {"name": "Joining", "desc": "**task_string_joined** = ','.join(task_list).", "hint": "`.join(list)`.", "solution": "task_string_joined = ','.join(task_list)", "check_var": "task_string_joined"},
            {"name": "String Formatting", "desc": '**invitation_message_formatted** = f"Dear {elite_guest_name}, you are cordially invited to the wedding."', "hint": "f-string.", "solution": 'invitation_message_formatted = f"Dear {elite_guest_name}, you are cordially invited to the wedding."', "check_var": "invitation_message_formatted"},
        ],
    },
]

# ============================
# Draft CSV helpers (after CATEGORIES)
# ============================

def _draft_path(roll_no: str) -> str:
    return os.path.join(DRAFTS_DIR, f"{roll_no}.csv")


def save_draft_csv(candidate: dict):
    if not candidate or not candidate.get("roll_no"):
        return
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roll_no": candidate.get("roll_no", ""),
        "name": candidate.get("name", ""),
        "email": candidate.get("email", ""),
        "section": candidate.get("section", ""),
        "about": candidate.get("about", ""),
        "cat_idx": st.session_state.get("cat_idx", 0),
        "task_idx": st.session_state.get("task_idx", 0),
        "submitted": int(bool(st.session_state.get("submitted", False))),
    }
    for cat in CATEGORIES:
        key = cat["key"]
        for i, _ in enumerate(cat["tasks"]):
            ck = f"code_{key}_{i}"
            row[ck] = st.session_state.answers.get(ck, "")
            row[f"rev_{key}_{i}"]  = int(st.session_state.progress[key]["revealed"][i])
            row[f"cor_{key}_{i}"]  = int(st.session_state.progress[key]["correct"][i])
            row[f"lock_{key}_{i}"] = int(st.session_state.progress[key]["locked"][i])
    pd.DataFrame([row]).to_csv(_draft_path(candidate["roll_no"]), index=False)


def load_draft_csv(roll_no: str):
    path = _draft_path(roll_no)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    rec = df.iloc[-1].to_dict()
    cand = {
        "roll_no": str(rec.get("roll_no", "")),
        "name": str(rec.get("name", "")),
        "email": str(rec.get("email", "")),
        "section": str(rec.get("section", "")),
        "about": str(rec.get("about", "")),
    }
    st.session_state.cat_idx = int(rec.get("cat_idx", 0))
    st.session_state.task_idx = int(rec.get("task_idx", 0))
    st.session_state.submitted = bool(rec.get("submitted", 0))
    for cat in CATEGORIES:
        key = cat["key"]
        n = len(cat["tasks"])
        for i in range(n):
            st.session_state.answers[f"code_{key}_{i}"] = str(rec.get(f"code_{key}_{i}", "") or "")
            st.session_state.progress[key]["revealed"][i] = bool(rec.get(f"rev_{key}_{i}", 0))
            st.session_state.progress[key]["correct"][i]  = bool(rec.get(f"cor_{key}_{i}", 0))
            st.session_state.progress[key]["locked"][i]   = bool(rec.get(f"lock_{key}_{i}", 0))
    return cand


def has_draft(roll_no: str) -> bool:
    return os.path.exists(_draft_path(roll_no))

# ============================
# Session state boot
# ============================
if "progress" not in st.session_state:
    st.session_state.progress = {}
if "cat_idx" not in st.session_state:
    st.session_state.cat_idx = 0
if "task_idx" not in st.session_state:
    st.session_state.task_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "candidate" not in st.session_state:
    st.session_state.candidate = None
if "task_select" not in st.session_state:
    st.session_state.task_select = 0
if "onboarding_step" not in st.session_state:
    st.session_state.onboarding_step = 0  # 0=welcome,1=personal,2=test

# Init progress structure per category
for cat in CATEGORIES:
    key = cat["key"]
    n = len(cat["tasks"])
    if key not in st.session_state.progress:
        st.session_state.progress[key] = {
            "revealed": [False]*n,
            "correct": [False]*n,
            "attempts": [0]*n,
            "locked": [False]*n,
        }
    else:
        cur = st.session_state.progress[key]
        for field, fill in (("revealed", False), ("correct", False), ("attempts", 0), ("locked", False)):
            lst = list(cur.get(field, []))
            if len(lst) < n:
                lst += [fill] * (n - len(lst))
            elif len(lst) > n:
                lst = lst[:n]
            cur[field] = lst

# ============================
# Auto-resume via URL param (?roll=...)
# ============================
qp = dict(st.query_params)
roll_qp = qp.get("roll")
def _persist_qp_with_roll(rno: str):
    st.query_params.update({"roll": rno})


if roll_qp and not st.session_state.candidate:
    cand = load_draft_csv(roll_qp)
    if cand:
        st.session_state.candidate = cand
        if has_submitted(roll_qp):
            st.session_state.submitted = True
        st.success(f"Welcome back, {cand['name']} — draft restored.")

# ============================
# Title
# ============================
st.title("🧪 ")

# ============================
# Onboarding Flow
# 0) Welcome screen
# 1) Personal details screen
# 2) Test screen
# ============================
if st.session_state.onboarding_step == 0:
    st.subheader("Welcome 👋")
    st.write("Please provide your basic details to continue.")
    with st.form("welcome_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="Mahima Gupta")
            roll_no = st.text_input("Roll No", placeholder="CS-2025-042")
        with col2:
            email = st.text_input("Email", placeholder="mahima@example.com")
            section = st.text_input("Class/Section", placeholder="CSE-A")
        agreed = st.checkbox("I confirm that I will submit only once.", value=False)
        go = st.form_submit_button("Let's Go ▶️")
    if go:
        if not (name and roll_no and agreed):
            st.error("Please fill Name, Roll No., and confirm the checkbox to continue.")
            st.stop()
        rno = roll_no.strip()
        if has_submitted(rno):
            st.error("Our records show you have already submitted this test. If this is a mistake, contact the instructor.")
            st.stop()
        st.session_state.candidate = {"name": name.strip(), "roll_no": rno, "email": email.strip(), "section": section.strip(), "about": ""}
        _persist_qp_with_roll(rno)
        if has_draft(rno):
            cand2 = load_draft_csv(rno)
            if cand2:
                st.session_state.candidate = cand2
                st.success("Draft found and restored.")
        save_draft_csv(st.session_state.candidate)
        st.session_state.onboarding_step = 1
        st.rerun()  # replaced experimental

elif st.session_state.onboarding_step == 1:
    st.subheader("Tell us a bit about you ✍️")
    with st.form("personal_form", clear_on_submit=False):
        about = st.text_input("Let's describe yourself in one line", placeholder="e.g., Python enthusiast who loves algorithms")
        nxt = st.form_submit_button("Continue to Test ▶️")
    if nxt:
        cand = st.session_state.candidate or {}
        cand["about"] = about.strip()
        st.session_state.candidate = cand
        save_draft_csv(cand)
        st.session_state.onboarding_step = 2
        st.rerun()  # replaced experimental

# Stop here if onboarding not complete
if st.session_state.onboarding_step < 2:
    st.stop()

candidate = st.session_state.candidate

# Helper: scoring
def total_tasks():
    return sum(len(c["tasks"]) for c in CATEGORIES)
def total_score():
    return sum(sum(1 for v in st.session_state.progress[c["key"]]["correct"] if v) for c in CATEGORIES)

# ============================
# LAYOUT: THREE COLUMNS
#   col1: Sections + Live Feedback
#   col2: Main Tasks / Exercises
#   col3: Scoreboard
# ============================
col1, col2, col3 = st.columns([0.25, 0.5, 0.25])

# ---------- LEFT: Sections + Live Feedback ----------
with col1:
    st.header("Sections")
    sel = st.radio(
        "Choose a data structure:",
        options=list(range(len(CATEGORIES))),
        format_func=lambda i: CATEGORIES[i]["title"],
        index=st.session_state.cat_idx,
        key="section_radio",
        disabled=st.session_state.submitted,
    )
    if sel != st.session_state.cat_idx:
        st.session_state.cat_idx = sel
        st.session_state.task_idx = 0
        st.session_state.task_select = 0
        save_draft_csv(candidate)

    st.divider()
    st.header("🟢 Live feedback")
    fb_text = st.text_area("Your feedback", height=90, placeholder="Share feedback in real time...")
    fb_rating = st.slider("Rating", min_value=1, max_value=5, value=5)
    if st.button("Submit feedback"):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roll_no": candidate.get("roll_no",""),
            "name": candidate.get("name",""),
            "rating": fb_rating,
            "feedback": fb_text.strip(),
        }
        if row["feedback"]:
            df = pd.DataFrame([row])
            if os.path.exists(FEEDBACK_CSV):
                df.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)
            else:
                df.to_csv(FEEDBACK_CSV, index=False)
            st.success("Thanks for the feedback!")
        else:
            st.warning("Please enter some feedback text before submitting.")
    st.caption("Recent feedback")
    if os.path.exists(FEEDBACK_CSV):
        try:
            df_fb = pd.read_csv(FEEDBACK_CSV)
            for _, r in df_fb.tail(5).iloc[::-1].iterrows():
                st.write(f"{r.get('timestamp','')} — {r.get('name','Anonymous')} ({r.get('rating','-')}/5): {r.get('feedback','')}")
        except Exception:
            st.write("No feedback yet.")
    else:
        st.write("No feedback yet.")

# ---------- MIDDLE: Main Tasks / Exercises ----------
with col2:
    cat = CATEGORIES[st.session_state.cat_idx]
    st.subheader(f"Section: {cat['title']}")

    def _sync_select_to_idx():
        st.session_state.task_idx = st.session_state.task_select
        save_draft_csv(candidate)

    def _go_next():
        names = [t["name"] for t in cat["tasks"]]
        st.session_state.task_idx = (st.session_state.task_idx + 1) % len(names)
        st.session_state.task_select = st.session_state.task_idx
        save_draft_csv(candidate)

    task_names = [t["name"] for t in cat["tasks"]]
    col_tl, col_tr = st.columns([0.8, 0.2])
    with col_tl:
        st.markdown("### Tasks")
        st.selectbox(
            "Jump to a task:",
            options=list(range(len(task_names))),
            format_func=lambda i: f"{i+1}. {task_names[i]}",
            index=st.session_state.task_idx,
            key="task_select",
            on_change=_sync_select_to_idx,
            disabled=st.session_state.submitted,
        )
    with col_tr:
        st.button("➜ Next task", on_click=_go_next, disabled=st.session_state.submitted)
        st.caption("Cycles to the first task after the last one.")

    idx = st.session_state.task_idx
    task = cat["tasks"][idx]
    cur_key = cat["key"]
    cur_prog = st.session_state.progress[cur_key]

    st.markdown("---")
    st.markdown(f"### 📝 {task['name']}")
    st.write(task["desc"])

    with st.expander("📦 Variables available to your code (initial state)", expanded=True):
        ns_preview, err = run_code(cat["init_code"], "")
        if err:
            st.error("Internal error in init code.")
        else:
            for var in cat["show_vars"]:
                if var in ns_preview:
                    st.code(f"{var} = {repr(ns_preview[var])}")

    with st.expander("💡 Hint"):
        st.write(task["hint"])

    def _autosave():
        save_draft_csv(candidate)

    code_key = f"code_{cur_key}_{idx}"
    existing = st.session_state.answers.get(code_key, "")
    locked = st.session_state.submitted or cur_prog["locked"][idx]
    user_code = st.text_area(
        "Your code:",
        value=existing,
        height=180,
        key=code_key,
        disabled=locked,
        on_change=_autosave,
    )
    st.session_state.answers[code_key] = st.session_state.get(code_key, existing)

    cols = st.columns([0.25, 0.25, 0.5])
    with cols[0]:
        run_clicked = st.button("✅ Check my answer", disabled=locked)
    with cols[1]:
        reveal_clicked = st.button("🕵️ Reveal solution", disabled=locked)

    if reveal_clicked and not locked:
        cur_prog["revealed"][idx] = True
        st.warning("You revealed the solution for this task. **No marks will be awarded** for this task even if you submit the correct answer later.")
        sol_ns, sol_err = run_code(cat["init_code"], task["solution"])
        st.markdown("**Standard solution:**")
        st.code(task["solution"])
        if not sol_err:
            target = task["check_var"]
            if target in sol_ns:
                st.markdown("**Expected value of target variable:**")
                st.code(f"{target} = {repr(sol_ns[target])}")
        save_draft_csv(candidate)

    if run_clicked and not locked:
        cur_prog["attempts"][idx] += 1
        sol_ns, sol_err = run_code(cat["init_code"], task["solution"])
        user_ns, user_err = run_code(cat["init_code"], st.session_state.answers.get(code_key, ""))

        if user_err:
            st.error("Your code raised an error:")
            st.code(user_err)
        elif sol_err:
            st.error("Internal error in solution code:")
            st.code(sol_err)
        else:
            target = task["check_var"]
            expected = sol_ns.get(target, None)
            got = (user_ns or {}).get(target, None)
            with st.expander("🔍 Your resulting variables (after running your code)"):
                for var in cat["show_vars"]:
                    if user_ns and var in user_ns:
                        st.code(f"{var} = {repr(user_ns[var])}")
                if got is not None:
                    st.code(f"{target} = {repr(got)}")
            if expected == got and not cur_prog["revealed"][idx]:
                if not cur_prog["correct"][idx]:
                    cur_prog["correct"][idx] = True
                st.success("Great job! Your answer matches the expected result. ✅ Point awarded.")
                st.balloons()
            else:
                cur_prog["locked"][idx] = True
                st.error("That's not correct. This task is now locked. See the correct solution below.")
                st.markdown("**Correct solution:**")
                st.code(task["solution"])
                st.markdown(f"**Expected `{target}` value:**")
                st.code(repr(expected))
        save_draft_csv(candidate)

    st.markdown("---")

    # Finalize & submit (CSV only, one-time)
    def evaluate_all():
        results = []
        score = 0
        revealed_count = 0
        attempted_count = 0
        correct_count = 0
        for cat in CATEGORIES:
            key = cat["key"]
            for i, task in enumerate(cat["tasks"]):
                code_k = f"code_{key}_{i}"
                code = st.session_state.answers.get(code_k, "")
                revealed = st.session_state.progress[key]["revealed"][i]
                locked = st.session_state.progress[key]["locked"][i]
                if revealed:
                    revealed_count += 1
                attempted = bool(code.strip())
                if attempted:
                    attempted_count += 1
                sol_ns, sol_err = run_code(cat["init_code"], task["solution"])
                user_ns, user_err = run_code(cat["init_code"], code)
                status = "unattempted"
                expected_val = None
                user_val = None
                if sol_err:
                    status = "error_ref"
                elif user_err:
                    status = "error_user"
                else:
                    target = task["check_var"]
                    if target in sol_ns:
                        expected_val = sol_ns[target]
                    if user_ns and target in user_ns:
                        user_val = user_ns[target]
                    if expected_val == user_val and not revealed:
                        status = "correct"
                        correct_count += 1
                        score += 1
                    else:
                        status = "wrong" if attempted else "unattempted"
                results.append({
                    "category": cat["title"],
                    "category_key": key,
                    "task_index": i+1,
                    "task_name": task["name"],
                    "attempted": attempted,
                    "revealed": revealed,
                    "locked": locked,
                    "status": status,
                    "check_var": task["check_var"],
                    "expected_repr": repr(expected_val),
                    "user_value_repr": repr(user_val),
                    "user_code": code,
                })
        return {
            "results": results,
            "score": score,
            "max_score": sum(len(c["tasks"]) for c in CATEGORIES),
            "revealed_count": revealed_count,
            "attempted_count": attempted_count,
            "correct_count": correct_count,
        }

    final_cols = st.columns([0.5, 0.5])
    with final_cols[0]:
        finalize = st.button("🧷 Finalize & Submit Test", type="primary", disabled=st.session_state.submitted)
    with final_cols[1]:
        st.caption("You can submit **once**. After submitting, editing is disabled and you'll see correct answers for any wrong items.")

    if finalize and not st.session_state.submitted:
        eval_pack = evaluate_all()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_row = {
            "timestamp": now,
            "roll_no": candidate["roll_no"],
            "name": candidate["name"],
            "email": candidate.get("email",""),
            "section": candidate.get("section",""),
            "about": candidate.get("about",""),
            "score": eval_pack["score"],
            "max_score": eval_pack["max_score"],
            "revealed_count": eval_pack["revealed_count"],
            "attempted_count": eval_pack["attempted_count"],
            "correct_count": eval_pack["correct_count"],
        }
        details_rows = []
        for r in eval_pack["results"]:
            details_rows.append({
                "timestamp": now,
                "roll_no": candidate["roll_no"],
                "name": candidate["name"],
                "category": r["category"],
                "category_key": r["category_key"],
                "task_index": r["task_index"],
                "task_name": r["task_name"],
                "attempted": r["attempted"],
                "revealed": r["revealed"],
                "locked": r["locked"],
                "status": r["status"],
                "check_var": r["check_var"],
                "expected_repr": r["expected_repr"],
                "user_value_repr": r["user_value_repr"],
                "user_code": r["user_code"],
            })
        # Save
        df_sum = pd.DataFrame([summary_row])
        if os.path.exists(SUMMARY_CSV):
            df_sum.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
        else:
            df_sum.to_csv(SUMMARY_CSV, index=False)
        if details_rows:
            df_det = pd.DataFrame(details_rows)
            if os.path.exists(DETAILS_CSV):
                df_det.to_csv(DETAILS_CSV, mode="a", header=False, index=False)
            else:
                df_det.to_csv(DETAILS_CSV, index=False)

        st.session_state.submitted = True
        save_draft_csv(candidate)
        st.success(f"Submitted! Score: {summary_row['score']} / {summary_row['max_score']}")

    if st.session_state.submitted:
        st.info("Submission locked. Review below — incorrect items show the correct answer.")
        eval_pack = evaluate_all()
        st.subheader(f"Result: {eval_pack['score']} / {eval_pack['max_score']}")
        for i, r in enumerate(eval_pack["results"], start=1):
            title = f"{r['category']} ▸ {r['task_index']}. {r['task_name']}"
            if r["status"] in ("wrong", "error_user"):
                with st.expander(f"❌ {title}"):
                    if r["user_code"]:
                        st.markdown("**Your code:**")
                        st.code(r["user_code"])
                    if r["user_value_repr"] is not None:
                        st.markdown(f"**Your `{r['check_var']}` value:**")
                        st.code(r["user_value_repr"])
                    sol_text = None
                    for c in CATEGORIES:
                        if c["title"] == r["category"]:
                            for t in c["tasks"]:
                                if t["name"] == r["task_name"]:
                                    sol_text = t["solution"]
                                    break
                    if sol_text:
                        st.markdown("**Correct solution:**")
                        st.code(sol_text)
                    st.markdown(f"**Expected `{r['check_var']}` value:**")
                    st.code(r["expected_repr"])
            elif r["status"] == "correct":
                with st.expander(f"✅ {title}"):
                    st.markdown("Correct ✔️")
            else:
                with st.expander(f"⚠️ {title} (not answered)"):
                    st.markdown("You didn't submit an answer for this task.")

# ---------- RIGHT: Scoreboard ----------
with col3:
    st.header("🏆 Scoreboard")
    done = total_score()
    total = total_tasks()
    st.metric("Points", f"{done} / {total}")
    st.progress(0 if total == 0 else done/total)

    cur_cat = CATEGORIES[st.session_state.cat_idx]
    prog = st.session_state.progress[cur_cat["key"]]
    section_done = sum(1 for v in prog["correct"] if v)
    section_revealed = sum(1 for v in prog["revealed"] if v)
    st.caption(f"Section '{cur_cat['title']}' — ✅ {section_done}/{len(cur_cat['tasks'])} • 🔎 {section_revealed} revealed")
