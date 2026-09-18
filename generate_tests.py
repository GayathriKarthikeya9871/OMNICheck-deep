# # import os
# # import zipfile

# # # Create nested test directories
# # os.makedirs("test_data/folder_A", exist_ok=True)
# # os.makedirs("test_data/folder_B", exist_ok=True)

# # # 1. The Rulebook (Empty Template)
# # with open("test_data/rulebook.csv", "w") as f:
# #     f.write("Employee Name, Expense Amount, Phone Status, ID Status, Policy Violation (Max $500)?\n")

# # # 2. Fraudulent Document (Folder A)
# # with open("test_data/folder_A/receipt_01.txt", "w") as f:
# #     f.write("Employee: John Doe\nAmount: $7,500\nDate: 2026-09-14\nContact: +910000000000\nGov ID: 123456789012\nNote: Please approve ASAP.")

# # # 3. Valid Document (Folder B)
# # with open("test_data/folder_B/receipt_02.txt", "w") as f:
# #     f.write("Employee: Jane Smith\nAmount: $120.00\nDate: 2026-09-15\nContact: +14155552671\nNote: Client dinner.")

# # # Zip them up to test the recursive unpacker
# # with zipfile.ZipFile("mass_audit_test.zip", "w") as z:
# #     for root, dirs, files in os.walk("test_data"):
# #         for file in files:
# #             z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "test_data"))

# # print("✅ SUCCESS: Created 'mass_audit_test.zip' and 'test_data/rulebook.csv'!")


# import os
# import zipfile

# os.makedirs("enterprise_test_data/employee_submissions", exist_ok=True)
# os.makedirs("enterprise_test_data/vendor_invoices", exist_ok=True)

# with open("enterprise_test_data/policy_template.csv", "w") as f:
#     f.write("Entity Name, Billed Amount, Paid Amount, Variance, ID Status, Phone Status, Risk Level\n")

# with open("enterprise_test_data/vendor_invoices/invoice_101.txt", "w") as f:
#     f.write("Vendor: TechCorp\nBilled Amount: $5,000\nDate: 2026-10-01\nNote: For server maintenance.")
    
# with open("enterprise_test_data/employee_submissions/receipt_101.txt", "w") as f:
#     f.write("Vendor: TechCorp\nPaid Amount: $4,500\nDate: 2026-10-02\nNote: Payment for invoice 101. Negotiated a $500 discount.")

# with open("enterprise_test_data/employee_submissions/contractor_fraud.txt", "w") as f:
#     f.write("Contractor: Ghost LLC\nBilled Amount: $2,000\nPaid Amount: $2,000\nContact: +919999999999\nGov ID: 999988887777\nNote: ID is mathematically fake.")

# with open("enterprise_test_data/employee_submissions/expense_404.txt", "w") as f:
#     f.write("Employee: Alice\nBilled Amount: $150\nPaid Amount: $150\nContact: +14155552671\nNote: Dinner. See attached receipt image (receipt_img_404.jpg) for proof.")

# with zipfile.ZipFile("ultimate_stress_test.zip", "w") as z:
#     for root, dirs, files in os.walk("enterprise_test_data"):
#         for file in files:
#             z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "enterprise_test_data"))

# print("✅ SUCCESS: Created 'ultimate_stress_test.zip' and 'enterprise_test_data/policy_template.csv'!")

import os
import zipfile

os.makedirs("micro_test", exist_ok=True)

with open("micro_test/receipt.txt", "w") as f:
    f.write("Vendor: TinyCoffee\nAmount: $4.50\nDate: 2026-09-18")

with open("micro_test/rulebook.csv", "w") as f:
    f.write("Vendor Name, Amount, Date\n")

with zipfile.ZipFile("micro_test.zip", "w") as z:
    for root, dirs, files in os.walk("micro_test"):
        for file in files:
            z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "micro_test"))

print("✅ SUCCESS: Created 'micro_test.zip' and 'micro_test/rulebook.csv'!")