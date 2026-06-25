//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  amg_linklist.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

#define LIST_HEAD -1
#define LIST_TAIL -2

/*!
 * \fn void jx_dispose_elt
 * \brief Dispose of memory space used by the element
 *        pointed to by element_ptr.  Use the 'free()'
 *        system call to return it to the free memory pool.
 * \date 2011/09/03
 */
void 
jx_dispose_elt( jx_LinkList element_ptr )
{
   free( element_ptr );
}


/*!
 * \fn void jx_remove_point
 * \brief Removes a point from the lists.
 * \date 2011/09/03
 */ 
void 
jx_remove_point( jx_LinkList   *LoL_head_ptr, 
                 jx_LinkList   *LoL_tail_ptr, 
                 JX_Int            measure,
                 JX_Int            index, 
                 JX_Int           *lists, 
                 JX_Int           *where )
{
   jx_LinkList   LoL_head = *LoL_head_ptr;
   jx_LinkList   LoL_tail = *LoL_tail_ptr;
   jx_LinkList   list_ptr;

   list_ptr = LoL_head;

   do
   {
      if (measure == list_ptr->data)
      {
         /* point to be removed is only point on list, which must be destroyed */
         if (list_ptr->head == index && list_ptr->tail == index)
         {
            /* removing only list, so num_left better be 0! */
            if (list_ptr == LoL_head && list_ptr == LoL_tail)
            {
               LoL_head = NULL;
               LoL_tail = NULL;
               jx_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_head == list_ptr)   /* removing 1st (max_measure) list */
            {
               list_ptr -> next_elt -> prev_elt = NULL;
               LoL_head = list_ptr->next_elt;
               jx_dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_tail == list_ptr)    /* removing last list */
            {
               list_ptr -> prev_elt -> next_elt = NULL;
               LoL_tail = list_ptr->prev_elt;
               jx_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else
            {
               list_ptr -> next_elt -> prev_elt = list_ptr -> prev_elt;
               list_ptr -> prev_elt -> next_elt = list_ptr -> next_elt;
               jx_dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
         }
         else if (list_ptr->head == index)      /* index is head of list */
         {
            list_ptr->head = lists[index];
            where[lists[index]] = LIST_HEAD;
            return;
         }
         else if (list_ptr->tail == index)      /* index is tail of list */
         {
            list_ptr->tail = where[index];
            lists[where[index]] = LIST_TAIL;
            return;
         }
         else                                   /* index is in middle of list */
         {
            lists[where[index]] = lists[index];
            where[lists[index]] = where[index];
            return;
         }
      }
      list_ptr = list_ptr -> next_elt;
   } while (list_ptr != NULL);
   
   jx_printf("No such list!\n");
   return;
}


/*!
 * \fn jx_LinkList jx_create_elt
 * \brief Create an element using Item for its data field.
 * \date 2011/09/03
 */  
jx_LinkList 
jx_create_elt( JX_Int Item )
{
    jx_LinkList   new_elt_ptr;
 
   /*---------------------------------------------
    * Allocate memory space for the new node. 
    * return with error if no space available 
    *-------------------------------------------*/

    if ( (new_elt_ptr = (jx_LinkList) malloc (sizeof(jx_ListElement))) == NULL )
    {
       jx_printf("\n jx_create_elt: malloc failed \n\n");
    }
    else 
    {
       new_elt_ptr -> data = Item;
       new_elt_ptr -> next_elt = NULL;
       new_elt_ptr -> prev_elt = NULL;
       new_elt_ptr -> head = LIST_TAIL;
       new_elt_ptr -> tail = LIST_HEAD;
    }

    return (new_elt_ptr);
}


/*!
 * \fn void enter_on_lists
 * \brief places point in new list.
 * \date 2011/09/03
 */  
void 
jx_enter_on_lists( jx_LinkList   *LoL_head_ptr, 
                   jx_LinkList   *LoL_tail_ptr, 
                   JX_Int            measure,
                   JX_Int            index, 
                   JX_Int           *lists, 
                   JX_Int           *where )
{
   jx_LinkList   LoL_head = *LoL_head_ptr;
   jx_LinkList   LoL_tail = *LoL_tail_ptr;

   jx_LinkList   list_ptr;
   jx_LinkList   new_ptr;

   JX_Int old_tail;

   list_ptr = LoL_head;

   if (LoL_head == NULL)   /* no lists exist yet */
   {
      new_ptr = jx_create_elt(measure);
      new_ptr->head = index;
      new_ptr->tail = index;
      lists[index] = LIST_TAIL;
      where[index] = LIST_HEAD; 
      LoL_head = new_ptr;
      LoL_tail = new_ptr;

      *LoL_head_ptr = LoL_head;
      *LoL_tail_ptr = LoL_tail;
      return;
   }
   else
   {
      do
      {
         if (measure > list_ptr->data)
         {
            new_ptr = jx_create_elt(measure);
            new_ptr->head = index;
            new_ptr->tail = index;
            lists[index] = LIST_TAIL;
            where[index] = LIST_HEAD;

            if ( list_ptr->prev_elt != NULL)
            { 
               new_ptr->prev_elt            = list_ptr->prev_elt;
               list_ptr->prev_elt->next_elt = new_ptr;   
               list_ptr->prev_elt           = new_ptr;
               new_ptr->next_elt            = list_ptr;
            }
            else
            {
               new_ptr->next_elt  = list_ptr;
               list_ptr->prev_elt = new_ptr;
               new_ptr->prev_elt  = NULL;
               LoL_head = new_ptr;
            }

            *LoL_head_ptr = LoL_head;
            *LoL_tail_ptr = LoL_tail; 
            
            return;
         }
         else if (measure == list_ptr->data)
         {
            old_tail = list_ptr->tail;
            lists[old_tail] = index;
            where[index] = old_tail;
            lists[index] = LIST_TAIL;
            list_ptr->tail = index;
            return;
         }
      
         list_ptr = list_ptr->next_elt;
         
      } while (list_ptr != NULL);

      new_ptr = jx_create_elt(measure);   
      new_ptr->head = index;
      new_ptr->tail = index;
      lists[index] = LIST_TAIL;
      where[index] = LIST_HEAD;
      LoL_tail->next_elt = new_ptr;
      new_ptr->prev_elt = LoL_tail;
      new_ptr->next_elt = NULL;
      LoL_tail = new_ptr;

      *LoL_head_ptr = LoL_head;
      *LoL_tail_ptr = LoL_tail;
      
      return;
   }
}
