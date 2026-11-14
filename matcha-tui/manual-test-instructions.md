# Manual Test Instructions - Race Condition Fix

## Setup

Test files are located at:
- Receptor: `/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_protein_std.pdb`
- Ligand: `/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_ligand.sdf`

## Test 1: Single Task with Background/Resume

### Steps:
1. Start the app:
   ```bash
   cd /home/nikolenko/work/Projects/Matcha/matcha-tui
   npm start
   ```

2. Create first task:
   - Press `s` to start setup
   - Select receptor: Navigate to `/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_protein_std.pdb`
   - Select ligand: Navigate to `/home/nikolenko/work/Projects/Matcha/test/1HVY_D16_ligand.sdf`
   - Box setup: Press Enter (default blind docking)
   - Parameters: Set run name to "test_task_1", nSamples to 10 (for quick test)
   - Review: Press Enter to start

3. **Send to background:**
   - Wait for task to start running (you should see "Starting docking...")
   - Press `h` or `Esc` to send to background
   - You should return to Welcome screen

4. **View Jobs screen:**
   - Press `j` to open Jobs screen
   - ✅ **VERIFY:** You should see 1 job listed with:
     - Status: running
     - Run name: "test_task_1"
     - Progress: showing percentage
     - Mode: single

5. **Resume task:**
   - Press Enter on the job to resume
   - ✅ **VERIFY:** You should see the Running screen with progress
   - ✅ **VERIFY:** Check logs for "Resuming job ..." message
   - ✅ **VERIFY:** Progress should continue updating

6. Send to background again (press `h`)

### Expected Results:
- ✅ Task should continue running in background
- ✅ Jobs screen shows the active task
- ✅ Can resume and see progress
- ✅ Log should show "Resuming existing job..."

---

## Test 2: Multiple Tasks

### Steps:
1. Start the app (if not already running)

2. Create first task:
   - Press `s` to start setup
   - Select files, box, params
   - Set run name to "multi_task_1"
   - Start task

3. **Send first task to background:**
   - Press `h` immediately after task starts
   - Return to Welcome screen

4. Create second task:
   - Press `s` again
   - Select SAME files (or different ones)
   - Set run name to "multi_task_2"
   - Start task

5. **Send second task to background:**
   - Press `h`

6. **View Jobs screen:**
   - Press `j`
   - ✅ **CRITICAL VERIFY:** Both tasks should be visible!
   - ✅ Task 1: run name "multi_task_1", status running
   - ✅ Task 2: run name "multi_task_2", status running

7. **Switch between tasks:**
   - Use arrow keys to select task 1
   - Press Enter to view
   - Press `h` to background
   - Press `j` to view jobs
   - Select task 2
   - Press Enter to view
   - ✅ **VERIFY:** Both tasks maintain their own progress

### Expected Results:
- ✅ Both tasks visible in Jobs screen
- ✅ Each task has correct run name and config
- ✅ Can switch between tasks
- ✅ Progress is maintained independently

---

## Test 3: Task Completion

### Steps:
1. Let a task complete (or wait for one to finish)

2. Check Jobs screen:
   - Press `j`
   - ✅ **VERIFY:** Completed task shows status: "completed"
   - ✅ **VERIFY:** Progress shows 100%

3. Check Results screen:
   - Navigate to completed task
   - ✅ **VERIFY:** Results are accessible

---

## Logs to Check

While testing, check the logs for these messages:

### Starting new task:
```
Starting docking with config: ...
Backend initialized...
Starting new docking job...
Received job_id: <uuid>
Adding job to activeJobs with config: {...}
Job added to activeJobs, total jobs: 1
Job started with ID: <uuid>
```

### Resuming task:
```
Resuming existing job...
Resuming job <uuid>, not starting new docking job
Restored job <uuid>, status: running
```

### If something is wrong:
```
WARNING: pendingJobConfig is null, job NOT added to activeJobs!
```

---

## What Fixed

### Before Fix:
- ❌ Creating second task would lose first task
- ❌ `activeJobs` Map was always empty
- ❌ Jobs screen showed "No active jobs"
- ❌ Returning to running task showed no progress

### After Fix:
- ✅ `pendingJobConfig` snapshot captured before starting
- ✅ Job added to `activeJobs` Map when job_id received
- ✅ Progress synced to `activeJobs` Map
- ✅ Jobs screen shows all active tasks
- ✅ Can resume tasks and see progress
- ✅ Multiple tasks work independently

---

## Cleanup

After testing:
```bash
rm -rf /home/nikolenko/work/Projects/Matcha/matcha-tui/results/test_*
rm -rf /home/nikolenko/work/Projects/Matcha/matcha-tui/results/multi_*
```
