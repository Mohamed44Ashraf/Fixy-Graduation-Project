# *************************************************** version four
import numpy as np 
import pandas as pd 
import random 
from scipy.stats import truncnorm 

# ---------------- SETTINGS ---------------- 
RNG_SEED = 42 
np.random.seed(RNG_SEED) 
random.seed(RNG_SEED) 

N = 300000 
TECHNICIAN_COUNT = 20000
CUSTOMER_COUNT = 120000

# ---------------- SERVICE TYPES ---------------- 
service_type_list = [ 
    "plumbing","electrical","ac_repair","carpentry","cleaning", 
    "appliance_repair","painting","furniture_assembly","pest_control", 
    "locksmith","satellite_installation","it_support","roof_leak_repair", 
    "furniture_moving","delivery" 
] 

base = np.array([0.10,0.09,0.08,0.08,0.07,0.07,0.06,0.06,0.05,0.05,0.06,0.06,0.06,0.06,0.08])
noise = np.random.dirichlet(alpha=np.ones(len(base)) * 6) * 0.05 
service_probs = base + noise 
service_probs = service_probs / service_probs.sum() 

service_counts = np.maximum(1, (service_probs * TECHNICIAN_COUNT).astype(int)) 
technician_service_map = {} 
tid = 1 
for svc, cnt in zip(service_type_list, service_counts): 
    for _ in range(cnt): 
        technician_service_map[tid] = svc 
        tid += 1 
while tid <= TECHNICIAN_COUNT: 
    technician_service_map[tid] = np.random.choice(service_type_list, p=service_probs) 
    tid += 1 

# ---------------- GENERATE ROWS ---------------- 
technician_ids = [] 
service_per_row = [] 
for _ in range(N): 
    t = random.randint(1, TECHNICIAN_COUNT) 
    technician_ids.append(t) 
    service_per_row.append(technician_service_map.get(t, np.random.choice(service_type_list, p=service_probs))) 

customer_ids = np.random.randint(1, CUSTOMER_COUNT + 1, size=N) 


# DAY/TIME/WEATHER 
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] 
days_props = [0.16,0.16,0.16,0.16,0.10,0.10,0.16] 
day_of_week = np.random.choice(days, size=N, p=np.array(days_props)/sum(days_props)) 

time_of_day_list = ["morning","afternoon","evening","night"] 
time_probs = [0.34,0.36,0.25,0.05] 
time_of_day = np.random.choice(time_of_day_list, size=N, p=time_probs) 

weather_list = ["sunny","cloudy","rainy","stormy","hot"] 
weather_probs = [0.62,0.20,0.10,0.04,0.04] 
weather_condition = np.random.choice(weather_list, size=N, p=weather_probs) 

# ---------------- JOB DURATION ---------------- 
duration_specs = { 
    "plumbing": (30,150,75), "electrical": (30,180,80), "ac_repair": (60,300,140), 
    "carpentry": (30,360,120), "cleaning": (20,120,60), "appliance_repair": (30,180,70), 
    "painting": (60,480,180), "furniture_assembly": (30,240,90), "pest_control": (45,180,85), 
    "locksmith": (15,90,35), "satellite_installation": (60,240,100), "it_support": (15,180,50), 
    "roof_leak_repair": (60,360,140), "furniture_moving": (30,300,120), "delivery": (10,60,25) 
} 

def truncated_normal(size, mean, std, low, high): 
    a, b = (low - mean) / std, (high - mean) / std 
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size).astype(float) 

service_arr = np.array(service_per_row) 
duration = np.zeros(N, dtype=float) 
for svc in duration_specs: 
    mask = (service_arr == svc) 
    low, high, mean = duration_specs[svc] 
    std = max(5, (high - low) / 6) 
    if mask.sum() > 0: 
        duration[mask] = truncated_normal(mask.sum(), mean, std, low, high) 


outlier_mask = np.random.rand(N) < 0.012 
duration[outlier_mask] *= np.random.uniform(1.6, 3.0, size=outlier_mask.sum()) 
miss_mask = np.random.rand(N) < 0.006 
duration[miss_mask] = np.nan 

# ---------------- TECHNICIAN CLUSTERS & SKILL ---------------- 
cluster_probs = [0.22, 0.22, 0.28, 0.28]  
tech_clusters = np.random.choice([1,2,3,4], size=TECHNICIAN_COUNT+1, p=cluster_probs) 
cluster_skill_params = { 
    1: {'loc': 0.9, 'scale': 0.5},   
    2: {'loc': -0.4, 'scale': 0.8},  
    3: {'loc': 0.05, 'scale': 0.9}, 
    4: {'loc': -0.05, 'scale': 0.7} 
} 
tech_skill = np.zeros(TECHNICIAN_COUNT + 1) 
for t_id in range(1, TECHNICIAN_COUNT + 1): 
    cl = tech_clusters[t_id-1] 
    p = cluster_skill_params[cl] 
    tech_skill[t_id] = np.random.normal(loc=p['loc'], scale=p['scale']) 
tech_skill = (tech_skill - tech_skill.mean()) / (tech_skill.std() + 1e-9) 
base_skill_row = np.array([tech_skill[t] for t in technician_ids]) 
tech_cluster_row = np.array([tech_clusters[t-1] for t in technician_ids]) 

# ---------------- RESPONSE TIME ---------------- 
base_response = np.random.exponential(scale=18, size=N) + 4 
time_factor = np.array([1.0 if t != "night" else 1.35 for t in time_of_day]) 
weather_factor = np.array([1.0 if w in ["sunny","cloudy"] else 1.45 if w=="rainy" else 1.9 for w in weather_condition]) 
day_factor = np.array([1.1 if d in ["Friday","Saturday"] else 1.0 for d in day_of_week]) 
skill_response_mod = np.clip(1.0 - 0.08 * base_skill_row, 0.7, 1.3) 
response_time = base_response * time_factor * weather_factor * day_factor * skill_response_mod 
response_time = np.clip(response_time, 3, 300).astype(float) 
r_out = np.random.rand(N) < 0.018 
response_time[r_out] *= np.random.uniform(1.4, 3.5, size=r_out.sum()) 
response_missing = np.random.rand(N) < 0.008 
response_time[response_missing] = np.nan 

# ---------------- RATINGS ---------------- 
def to_int_rating(arr, na_prob=0.008): 
    r = np.round(arr).astype(float) 
    mask_na = np.random.rand(len(r)) < na_prob 
    r[mask_na] = np.nan 
    return np.clip(r, 1, 5) 

svc_mean = np.vectorize(lambda s: duration_specs[s][2])(service_arr) 
eff_ratio = np.nan_to_num(svc_mean / (duration + 1)) 

punct_base = 3.4 + 0.45 * base_skill_row - 0.0035 * np.nan_to_num(response_time, nan=60) / 10 
punctuality = to_int_rating(punct_base + np.random.normal(0, 0.45, N)) 

prof_base = 3.5 + 0.5 * base_skill_row + 0.55 * np.clip(eff_ratio - 1.0, -0.5, 1.0) 
professionalism = to_int_rating(prof_base + np.random.normal(0, 0.5, N)) 

comm_base = 3.3 + 0.42 * base_skill_row 
communication = to_int_rating(comm_base + np.random.normal(0, 0.45, N)) 

hq_base = np.where( 
    np.isin(service_arr, ["delivery","furniture_moving","furniture_assembly"]), 
    3.1 + 0.45 * base_skill_row + np.random.normal(0, 0.65, N), 
    3.5 + 0.45 * base_skill_row + np.random.normal(0, 0.55, N) 
) 
handling_quality = to_int_rating(hq_base) 

cost_base = 3.75 - 0.55 * np.clip((duration - svc_mean) / (svc_mean + 1), -0.8, 2.0) - 0.0018 * np.nan_to_num(response_time, 60) 
cost_satisfaction = to_int_rating(cost_base + np.random.normal(0, 0.55, N)) 

customer_punctuality = to_int_rating(3.6 + np.random.normal(0, 0.85, N)) 
customer_behavior = to_int_rating(3.45 + 0.48 * (cost_satisfaction - 3.5) + np.random.normal(0, 0.75, N)) 
clarity_of_issue = to_int_rating(3.55 + np.random.normal(0, 0.65, N)) 
safety_and_environment = to_int_rating(3.6 + np.random.normal(0, 0.6, N)) 

stack_cust = np.vstack([punctuality, professionalism, communication, handling_quality, cost_satisfaction]) 
cust_satisfaction = np.nanmean(stack_cust, axis=0) 
nan_idx = np.isnan(cust_satisfaction) 
if nan_idx.sum() > 0: 
    cust_satisfaction[nan_idx] = 3.2 + 0.35 * base_skill_row[nan_idx] + np.random.normal(0, 0.3, nan_idx.sum()) 
cust_satisfaction = np.round(np.clip(cust_satisfaction, 1, 5)).astype(int) 

stack_worker = np.vstack([customer_behavior, clarity_of_issue, safety_and_environment, customer_punctuality]) 
worker_satisfaction = np.nanmean(stack_worker, axis=0) 
nan_idx_w = np.isnan(worker_satisfaction) 
if nan_idx_w.sum() > 0: 
    worker_satisfaction[nan_idx_w] = 3.2 + 0.2 * base_skill_row[nan_idx_w] + np.random.normal(0, 0.3, nan_idx_w.sum()) 
worker_satisfaction = np.round(np.clip(worker_satisfaction, 1, 5)).astype(int) 

# ---------------- FINAL RATING BALANCED ---------------- 
perf_factor = 0.5 * (1 - np.tanh((duration - svc_mean) / (svc_mean + 1))) + 0.5 * (1 - np.tanh((response_time - 30) / 60)) 
perf_factor = np.nan_to_num(perf_factor, nan=0.0) 

difficulty_bonus = np.zeros(N) 
difficulty_bonus += np.where(np.isin(day_of_week, ["Saturday","Sunday"]), 0.12, 0.0) 
difficulty_bonus += np.where(time_of_day == "night", 0.06, 0.0) 
difficulty_bonus += np.where(np.isin(weather_condition, ["rainy","stormy","hot"]), 0.10, 0.0) 

w_cust, w_worker, w_perf, w_context = 0.55, 0.25, 0.15, 0.05 
base_combined = ( 
    w_cust * cust_satisfaction 
    + w_worker * worker_satisfaction 
    + w_perf * (1 + perf_factor * 1.8) 
    + w_context * (1 + difficulty_bonus * 1.5) 
) 


delta = worker_satisfaction - cust_satisfaction 
fairness = np.zeros(N) 
fairness += np.where(delta < -1.0, 0.30, 0.0) 
fairness += np.where((delta >= -1.0) & (delta < -0.5), 0.12, 0.0) 
fairness -= np.where(delta > 1.0, 0.20, 0.0) 

final_raw = base_combined + fairness 

# --- ترتيب و rescale للحصول على النسب المطلوبة ---
class_probs = {5: 0.35, 4:0.25, 3:0.20, 2:0.12, 1:0.08}
sorted_idx = np.argsort(final_raw)
final_rating_balanced = np.zeros(N, dtype=int)

start_idx = 0
for cls, prob in sorted(class_probs.items()):
    n_rows = int(N * prob)
    end_idx = start_idx + n_rows
    if cls == 5:  # آخر class ياخد الباقي
        end_idx = N
    final_rating_balanced[sorted_idx[start_idx:end_idx]] = cls
    start_idx = end_idx

final_rating = final_rating_balanced
satisfied = (final_rating >= 4).astype(int)





# ---------------- BUILD DATAFRAME ---------------- 
df = pd.DataFrame({ 
    "technician_id": technician_ids, 
    "customer_id": customer_ids, 
    "service_type": service_per_row, 
    "day_of_week": day_of_week, 
    "time_of_day": time_of_day, 
    "weather_condition": weather_condition, 
    "job_duration_minutes": duration, 
    "response_time_minutes": response_time, 
    "punctuality_rating": punctuality, 
    "professionalism_rating": professionalism, 
    "communication_rating": communication, 
    "handling_quality": handling_quality, 
    "cost_satisfaction": cost_satisfaction, 
    "customer_satisfaction": cust_satisfaction,  
    "customer_behavior": customer_behavior, 
    "clarity_of_issue": clarity_of_issue, 
    "safety_and_environment": safety_and_environment, 
    "customer_punctuality": customer_punctuality, 
    "worker_satisfaction": worker_satisfaction, 
    "final_rating": final_rating,
    "satisfied": satisfied
}) 



# ---------------- CLEAN RATING COLUMNS ---------------- 
rating_cols = [ 
    "punctuality_rating","professionalism_rating","communication_rating","handling_quality", 
    "customer_satisfaction","cost_satisfaction","customer_behavior","clarity_of_issue", 
    "safety_and_environment","customer_punctuality","worker_satisfaction","final_rating" 
] 
for col in rating_cols: 
    df[col] = pd.to_numeric(df[col], errors='coerce').round().fillna(3).astype(int) 
    df[col] = df[col].clip(1,5) 

# ---------------- SAVE CSV ---------------- 
OUT_FILE = "service_feedback_dataset_7.csv" 
df.to_csv(OUT_FILE, index=False) 
print("Saved CSV:", OUT_FILE) 
print("Shape:", df.shape) 




