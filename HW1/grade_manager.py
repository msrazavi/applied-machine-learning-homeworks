import csv
import os
import re

# UTILITY & HELPER FUNCTIONS
def validate_email(email):
    # Checks if email follows the format: user@domain.extension using Regular Expressions.
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if re.match(pattern, email):
        return True
    return False


def validate_grade(grade):
    # Checks if grade is a float between 0 and 100.
    try:
        g = float(grade)
        if 0.0 <= g <= 100.0:
            return True, g
    except ValueError:
        pass
    return False, None

def convert_to_letter_grade(average):
    # Converts numerical average to letter grade.
    if average >= 90: return 'A'
    elif average >= 80: return 'B'
    elif average >= 70: return 'C'
    elif average >= 60: return 'D'
    else: return 'F'

def convert_to_grade_points(letter):
    # Converts letter grade to GPA points.
    mapping = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    return mapping.get(letter, 0.0)

def display_menu():
    print("\n=== STUDENT GRADE MANAGEMENT SYSTEM ===")
    print("1. Add New Student")
    print("2. View All Students")
    print("3. Search Student")
    print("4. Update Student Information")
    print("5. Add New Course")
    print("6. View All Courses")
    print("7. Enroll Student in Course")
    print("8. Record Grade")
    print("9. Generate Student Report (GPA)")
    print("10. Generate Course Report (Average)")
    print("11. Save Data to File")
    print("12. Load Data from File")
    print("13. Export to CSV")
    print("0. Exit")
    print("=======================================")

# STUDENT MANAGEMENT FUNCTIONS
def add_student(students, student_id, name, email):
    if student_id in students:
        print(f"Error: Student ID {student_id} already exists.")
        return
    
    if not validate_email(email):
        print("Error: Invalid email format.")
        return

    students[student_id] = {
        "name": name,
        "email": email,
        "grades": {}
    }
    print(f"Student {name} added successfully!")

def display_all_students(students):
    if not students:
        print("No students found.")
        return
    
    print(f"\n{'ID':<10} {'Name':<20} {'Email':<30}")
    print("-" * 60)
    for sid, info in students.items():
        print(f"{sid:<10} {info['name']:<20} {info['email']:<30}")

def search_student(students, student_id):
    if student_id in students:
        info = students[student_id]
        print("\n--- Student Found ---")
        print(f"ID: {student_id}")
        print(f"Name: {info['name']}")
        print(f"Email: {info['email']}")
        print("Enrolled Courses:", ", ".join(info['grades'].keys()))
    else:
        print("Student not found.")

def update_student(students, student_id, new_name, new_email):
    if student_id not in students:
        print("Student not found.")
        return

    if new_name:
        students[student_id]['name'] = new_name
    if new_email:
        if validate_email(new_email):
            students[student_id]['email'] = new_email
        else:
            print("Invalid email format. Email not updated.")
            return
    print("Student information updated.")

# COURSE MANAGEMENT FUNCTIONS
def add_course(courses, course_code, course_name, credits):
    if course_code in courses:
        print("Error: Course code already exists.")
        return
    
    courses[course_code] = {
        "name": course_name,
        "credits": credits
    }
    print(f"Course {course_name} added successfully!")

def display_all_courses(courses):
    if not courses:
        print("No courses found.")
        return

    print(f"\n{'Code':<10} {'Name':<30} {'Credits':<10}")
    print("-" * 50)
    for code, info in courses.items():
        print(f"{code:<10} {info['name']:<30} {info['credits']:<10}")

def get_course_credits(courses, course_code):
    if course_code in courses:
        return courses[course_code]['credits']
    return 0

# GRADE MANAGEMENT FUNCTIONS
def enroll_student(students, student_id, course_code):
    if student_id not in students:
        print("Student not found.")
        return
    
    # We check validation against the courses dict in the main loop or here
    # For now, we assume the course code provided is valid or checked before calling
    if course_code in students[student_id]['grades']:
        print("Student is already enrolled in this course.")
    else:
        students[student_id]['grades'][course_code] = []
        print(f"Student enrolled in {course_code}.")

def record_grade(students, student_id, course_code, grade):
    if student_id not in students:
        print("Student not found.")
        return
    
    if course_code not in students[student_id]['grades']:
        print("Student is not enrolled in this course.")
        return

    valid, g_val = validate_grade(grade)
    if valid:
        students[student_id]['grades'][course_code].append(g_val)
        print("Grade recorded successfully.")
    else:
        print("Error: Grade must be a number between 0 and 100.")

def calculate_course_average(students, course_code):
    total_grades = []
    for s_info in students.values():
        if course_code in s_info['grades']:
            total_grades.extend(s_info['grades'][course_code])
    
    if not total_grades:
        return 0.0
    
    return sum(total_grades) / len(total_grades)

def calculate_student_gpa(students, courses, student_id):
    if student_id not in students:
        return 0.0
    
    total_points = 0
    total_credits = 0
    
    for course_code, grades in students[student_id]["grades"].items():
        # Only calculate if course exists in our course list and student has grades
        if course_code in courses and grades:
            course_avg = sum(grades) / len(grades)
            letter = convert_to_letter_grade(course_avg)
            points = convert_to_grade_points(letter)
            
            crs_credits = courses[course_code]["credits"]
            
            total_points += points * crs_credits
            total_credits += crs_credits
            
    if total_credits == 0:
        return 0.0
        
    return total_points / total_credits

# FILE OPERATIONS FUNCTIONS
def save_data_to_file(students, courses, filename):
    try:
        with open(filename, 'w') as file:
            # Save Courses
            file.write("COURSES\n")
            for code, info in courses.items():
                file.write(f"{code},{info['name']},{info['credits']}\n")
            
            # Save Students
            file.write("STUDENTS\n")
            for sid, info in students.items():
                file.write(f"{sid},{info['name']},{info['email']}\n")
            
            # Save Grades (Custom format: GRADES,student_id,course_code,grade1,grade2...)
            for sid, info in students.items():
                for code, grades_list in info['grades'].items():
                    if grades_list:
                        g_str = ",".join(map(str, grades_list))
                        file.write(f"GRADES,{sid},{code},{g_str}\n")
                    else:
                        # Save enrollment even if no grades yet
                        file.write(f"GRADES,{sid},{code}\n")
                        
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data_from_file(students, courses, filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        # Clear current data
        students.clear()
        courses.clear()
        
        mode = None # To track which section we are reading
        
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line: continue
                
                if line == "COURSES":
                    mode = "COURSES"
                    continue
                elif line == "STUDENTS":
                    mode = "STUDENTS"
                    continue
                elif line.startswith("GRADES"):
                    # Handle grades line immediately
                    parts = line.split(',')
                    # GRADES, sid, course_code, [grades...]
                    if len(parts) >= 3:
                        sid = parts[1]
                        code = parts[2]
                        
                        # Ensure student exists before adding grades (safety check)
                        if sid in students:
                            if code not in students[sid]['grades']:
                                students[sid]['grades'][code] = []
                            
                            # Add grades if they exist
                            for g in parts[3:]:
                                if g: # check if string is not empty
                                    students[sid]['grades'][code].append(float(g))
                    continue

                # Process specific section data
                if mode == "COURSES":
                    parts = line.split(',')
                    if len(parts) == 3:
                        courses[parts[0]] = {"name": parts[1], "credits": int(parts[2])}
                
                elif mode == "STUDENTS":
                    parts = line.split(',')
                    if len(parts) == 3:
                        students[parts[0]] = {
                            "name": parts[1], 
                            "email": parts[2], 
                            "grades": {}
                        }
        
        print(f"Data successfully loaded from {filename}")

    except Exception as e:
        print(f"Error loading file: {e}")

def export_grades_to_csv(students, courses, filename):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow(['Student ID', 'Name', 'Course Code', 'Course Name', 'Average', 'Letter Grade'])
            
            for sid, s_info in students.items():
                for code, grades in s_info['grades'].items():
                    course_name = "Unknown"
                    if code in courses:
                        course_name = courses[code]['name']
                    
                    if grades:
                        avg = sum(grades) / len(grades)
                        letter = convert_to_letter_grade(avg)
                        writer.writerow([sid, s_info['name'], code, course_name, round(avg, 2), letter])
                    else:
                        writer.writerow([sid, s_info['name'], code, course_name, "N/A", "N/A"])
        
        print(f"Report exported to {filename}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")

# MAIN PROGRAM
def main():
    # Initialize data structures
    students = {}
    courses = {}
    
    data_file = "data.txt"

    # Auto-load if file exists
    if os.path.exists(data_file):
        load_data_from_file(students, courses, data_file)

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            sid = input("Enter Student ID: ")
            name = input("Enter Name: ")
            email = input("Enter Email: ")
            add_student(students, sid, name, email)

        elif choice == "2":
            display_all_students(students)

        elif choice == "3":
            sid = input("Enter Student ID to Search: ")
            search_student(students, sid)

        elif choice == "4":
            sid = input("Enter Student ID to Update: ")
            name = input("Enter new name (press Enter to skip): ")
            email = input("Enter new email (press Enter to skip): ")
            update_student(students, sid, name, email)

        elif choice == "5":
            code = input("Enter Course Code: ")
            name = input("Enter Course Name: ")
            try:
                crd = int(input("Enter Credits: "))
                add_course(courses, code, name, crd)
            except ValueError:
                print("Error: Credits must be an integer.")

        elif choice == "6":
            display_all_courses(courses)

        elif choice == "7":
            sid = input("Enter Student ID: ")
            code = input("Enter Course Code: ")
            if code in courses:
                enroll_student(students, sid, code)
            else:
                print("Error: Course does not exist.")

        elif choice == "8":
            sid = input("Enter Student ID: ")
            code = input("Enter Course Code: ")
            grade = input("Enter Grade (0-100): ")
            record_grade(students, sid, code, grade)

        elif choice == "9":
            sid = input("Enter Student ID for Report: ")
            if sid in students:
                gpa = calculate_student_gpa(students, courses, sid)
                print(f"\nReport for {students[sid]['name']}")
                print(f"GPA: {gpa:.2f}")
            else:
                print("Student not found.")

        elif choice == "10":
            code = input("Enter Course Code for Report: ")
            if code in courses:
                avg = calculate_course_average(students, code)
                print(f"\nAverage for {courses[code]['name']}: {avg:.2f}")
            else:
                print("Course not found.")

        elif choice == "11":
            save_data_to_file(students, courses, data_file)

        elif choice == "12":
            load_data_from_file(students, courses, data_file)

        elif choice == "13":
            filename = input("Enter filename for export (e.g., report.csv): ")
            export_grades_to_csv(students, courses, filename)

        elif choice == "0":
            save = input("Save changes before exit? (y/n): ")
            if save.lower() == 'y':
                save_data_to_file(students, courses, data_file)
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
