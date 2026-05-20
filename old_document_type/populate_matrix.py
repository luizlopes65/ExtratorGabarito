#!/usr/bin/env python3
"""
Populate the matriz_assuntos_subatributos.csv with random question numbers.

This script reads the empty matrix and populates cells randomly with question
numbers from 1 to 103. Multiple questions can appear in the same cell.
"""

import csv
import random
from pathlib import Path


def populate_matrix_randomly(
    input_file: str = "matriz_assuntos_subatributos.csv",
    output_file: str = "matriz_assuntos_subatributos_populated.csv",
    max_question: int = 103,
    fill_probability: float = 0.3,  # 30% chance to fill a cell
    max_questions_per_cell: int = 3
):
    """
    Populate matrix with random question numbers.
    
    Args:
        input_file: Input matrix CSV file
        output_file: Output populated matrix CSV file
        max_question: Maximum question number (1 to max_question)
        fill_probability: Probability of filling each cell (0.0 to 1.0)
        max_questions_per_cell: Maximum number of questions per cell
    """
    
    # Read the matrix
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Populate cells randomly
    populated_rows = []
    total_cells = 0
    filled_cells = 0
    total_questions_assigned = 0
    
    for row in rows:
        new_row = {'Assunto': row['Assunto']}
        
        for col in fieldnames[1:]:  # Skip 'Assunto' column
            total_cells += 1
            
            # Decide if we should fill this cell
            if random.random() < fill_probability:
                # Decide how many questions to put in this cell
                num_questions = random.randint(1, max_questions_per_cell)
                
                # Generate random question numbers
                questions = random.sample(range(1, max_question + 1), num_questions)
                questions.sort()
                
                # Format as comma-separated string
                new_row[col] = ', '.join(str(q) for q in questions)
                filled_cells += 1
                total_questions_assigned += num_questions
            else:
                new_row[col] = ''
        
        populated_rows.append(new_row)
    
    # Write the populated matrix
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(populated_rows)
    
    print(f"✓ Matrix populated successfully!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"\nStatistics:")
    print(f"  Total cells: {total_cells}")
    print(f"  Filled cells: {filled_cells} ({filled_cells/total_cells*100:.1f}%)")
    print(f"  Empty cells: {total_cells - filled_cells} ({(total_cells-filled_cells)/total_cells*100:.1f}%)")
    print(f"  Total questions assigned: {total_questions_assigned}")
    print(f"  Average questions per filled cell: {total_questions_assigned/filled_cells:.2f}")


if __name__ == '__main__':
    populate_matrix_randomly()

 
