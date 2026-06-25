//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_parser_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

typedef struct _jx_optionsNode jx_OptionsNode;

struct _jx_parser_dh
{
    jx_OptionsNode *head;
    jx_OptionsNode *tail;
};

struct _jx_optionsNode
{
    char *name;
    char *value;
    jx_OptionsNode *next;
};

static jx_bool jx_find( jx_Parser_dh p, char *option, jx_OptionsNode** ptr );
static void jx_init_from_default_settings_private( jx_Parser_dh p );

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhCreate"
void jx_Parser_dhCreate( jx_Parser_dh *p )
{
    JX_START_FUNC_DH
    jx_OptionsNode *ptr;
    struct _jx_parser_dh *tmp = (struct _jx_parser_dh *)JX_MALLOC_DH(sizeof(struct _jx_parser_dh)); JX_CHECK_V_ERROR;
   *p = tmp;
    tmp->head = tmp->tail = (jx_OptionsNode*)JX_MALLOC_DH(sizeof(jx_OptionsNode)); JX_CHECK_V_ERROR;
    ptr = tmp->head;
    ptr->next = NULL;
    ptr->name  = (char *)JX_MALLOC_DH(6*sizeof(char)); JX_CHECK_V_ERROR;
    ptr->value = (char *)JX_MALLOC_DH(6*sizeof(char)); JX_CHECK_V_ERROR;
    strcpy(ptr->name, "JUNK");
    strcpy(ptr->value, "JUNK");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhDestroy"
void jx_Parser_dhDestroy( jx_Parser_dh p )
{
    JX_START_FUNC_DH
    jx_OptionsNode *ptr2 = p->head, *ptr1 = ptr2;
    if (ptr1 != NULL)
    {
        do
        {
            ptr2 = ptr2->next;
            JX_FREE_DH(ptr1->name);
            JX_FREE_DH(ptr1->value);
            JX_FREE_DH(ptr1);
            ptr1 = ptr2;
        } while (ptr1 != NULL);
    }
    JX_FREE_DH(p);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhUpdateFromFile"
void jx_Parser_dhUpdateFromFile( jx_Parser_dh p, char *filename )
{
    JX_START_FUNC_DH_2
    char line[80], name[80], value[80];
    FILE *fp;
    
    if ((fp = fopen(filename, "r")) == NULL)
    {
        jx_sprintf(jx_msgBuf_dh, "can't open >>%s<< for reading", filename);
        JX_SET_INFO(jx_msgBuf_dh);
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "updating parser from file: >>%s<<", filename);
        JX_SET_INFO(jx_msgBuf_dh);
        while (!feof(fp))
        {
            if (fgets(line, 80, fp) == NULL) break;
            if (line[0] != '#')
            {
                if (jx_sscanf(line, "%s %s", name, value) != 2) break;
                jx_Parser_dhInsert(p, name, value);
            }
        }
        fclose(fp);
    }
    JX_END_FUNC_DH_2
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhInit"
void jx_Parser_dhInit( jx_Parser_dh p, JX_Int argc, char *argv[] )
{
    JX_START_FUNC_DH_2
    JX_Int j;
    
    jx_init_from_default_settings_private(p); JX_CHECK_V_ERROR;
    jx_Parser_dhUpdateFromFile(p, "./database"); JX_CHECK_V_ERROR;
    for (j = 1; j < argc; ++ j)
    {
        if (strcmp(argv[j],"-db_filename") == 0)
        {
            ++ j;
            if (j < argc)
            {
                jx_Parser_dhUpdateFromFile(p, argv[j]); JX_CHECK_V_ERROR;
            }
        }
    }
    JX_Int i = 0;
    while (i < argc)
    {
        if (argv[i][0] == '-')
        {
            char value[] = {"1"};
            jx_bool flag = jx_false;
            
            if (i+1 < argc && argv[i+1][0] == '-' && argv[i+1][1] == '-')
            {
                flag = jx_true;
            }
            if ((i+1 == argc || argv[i+1][0] == '-') && !flag)
            {
                jx_Parser_dhInsert(p, argv[i], value);
            }
            else if (flag)
            {
                jx_Parser_dhInsert(p, argv[i], argv[i+1]+1);
            }
            else
            {
                jx_Parser_dhInsert(p, argv[i], argv[i+1]);
            }
        }
        ++ i;
    }
    JX_END_FUNC_DH_2
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhHasSwitch"
jx_bool jx_Parser_dhHasSwitch( jx_Parser_dh p, char *s )
{
    JX_START_FUNC_DH_2
    jx_bool has_switch = jx_false;
    jx_OptionsNode *node;
    
    if (p != NULL && jx_find(p,s,&node))
    {
        if (!strcmp(node->value, "0"))
        {
            has_switch = jx_false;
        }
        else if (!strcmp(node->value, "jx_false"))
        {
            has_switch = jx_false;
        }
        else if (!strcmp(node->value, "False"))
        {
            has_switch = jx_false;
        }
        else if (!strcmp(node->value, "FALSE"))
        {
            has_switch = jx_false;
        }
        else
        {
            has_switch = jx_true;
        }
    }
    JX_END_FUNC_VAL_2(has_switch)
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhReadInt"
jx_bool jx_Parser_dhReadInt( jx_Parser_dh p, char *in, JX_Int *out )
{
    JX_START_FUNC_DH_2
    jx_bool has_switch = jx_false;
    jx_OptionsNode *node;
    
    if (p != NULL && jx_find(p,in,&node))
    {
       *out = atoi(node->value);
        if (!strcmp(node->value, "0"))
        {
            has_switch = jx_false;
        }
        else
        {
            has_switch = jx_true;
        }
    }
    JX_END_FUNC_VAL_2(has_switch)
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhReadDouble"
jx_bool jx_Parser_dhReadDouble( jx_Parser_dh p, char *in, JX_Real *out )
{
    JX_START_FUNC_DH_2
    jx_bool optionExists = jx_false;
    jx_OptionsNode *node;
    
    if (p != NULL && jx_find(p,in,&node))
    {
       *out = atof(node->value);
        optionExists = jx_true;
    }
    JX_END_FUNC_VAL_2(optionExists)
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhReadString"
jx_bool jx_Parser_dhReadString( jx_Parser_dh p, char *in, char **out )
{
    JX_START_FUNC_DH_2
    jx_bool optionExists = jx_false;
    jx_OptionsNode *node;
    
    if (p != NULL && jx_find(p,in,&node))
    {
       *out = node->value;
        optionExists = jx_true;
    }
    JX_END_FUNC_VAL_2(optionExists)
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhPrint"
void jx_Parser_dhPrint( jx_Parser_dh p, FILE *fp, jx_bool allPrint )
{
    JX_START_FUNC_DH_2
    jx_OptionsNode *ptr = p->head;
    
    if (fp == NULL) JX_SET_V_ERROR("fp == NULL");
    if (jx_myid_dh == 0 || allPrint)
    {
        jx_fprintf(fp, "------------------------ registered options:\n");
        if (ptr == NULL)
        {
            jx_fprintf(fp, "Parser object is invalid; nothing to print!\n");
        }
        else
        {
            ptr = ptr->next;
            while (ptr != NULL)
            {
                jx_fprintf(fp, "   %s  %s\n", ptr->name, ptr->value);
                fflush(fp);
                ptr = ptr->next;
            }
        }
        jx_fprintf(fp, "\n");
        fflush(fp);
    }
    JX_END_FUNC_DH_2
}

#undef __FUNC__
#define __FUNC__ "jx_Parser_dhInsert"
void jx_Parser_dhInsert( jx_Parser_dh p, char *option, char *value )
{
    JX_START_FUNC_DH_2
    jx_OptionsNode *node;
    JX_Int length;
    
    if (p == NULL) goto PARSER_NOT_INITED;
    if (jx_find(p,option,&node))
    {
        JX_Int length2 = strlen(node->value) + 1;
        
        length = strlen(value) + 1;
        if (length2 < length)
        {
            JX_FREE_DH(node->value);
            node->value  = (char *)JX_MALLOC_DH(length*sizeof(char)); JX_CHECK_V_ERROR;
        }
        strcpy(node->value, value);
    }
    else
    {
        node = p->tail;
        p->tail = node->next = (jx_OptionsNode *)JX_MALLOC_DH(sizeof(jx_OptionsNode)); JX_CHECK_V_ERROR;
        node = node->next;
        length = strlen(option) + 1;
        node->name = (char *)JX_MALLOC_DH(length*sizeof(char)); JX_CHECK_V_ERROR;
        strcpy(node->name, option);
        length = strlen(value) + 1;
        node->value = (char *)JX_MALLOC_DH(length*sizeof(char)); JX_CHECK_V_ERROR;
        strcpy(node->value, value);
        node->next = NULL;
    }
    
PARSER_NOT_INITED: ;
    
    JX_END_FUNC_DH_2
}

#undef __FUNC__
#define __FUNC__ "jx_find"
jx_bool jx_find( jx_Parser_dh p, char *option, jx_OptionsNode **ptr )
{
    JX_START_FUNC_DH_2
    jx_OptionsNode *tmpPtr = p->head;
    jx_bool foundit = jx_false;
    
    while (tmpPtr != NULL)
    {
        if (strcmp(tmpPtr->name,option) == 0)
        {
            foundit = jx_true;
           *ptr = tmpPtr;
            break;
        }
        tmpPtr = tmpPtr->next;
    }
    JX_END_FUNC_VAL_2(foundit)
}

#undef __FUNC__
#define __FUNC__ "jx_init_from_default_settings_private"
void jx_init_from_default_settings_private( jx_Parser_dh p )
{
    JX_START_FUNC_DH_2
    jx_Parser_dhInsert(p, "-sig_dh", "1"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-px", "1"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-py", "1"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-pz", "0"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-m", "4"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-xx_coeff", "-1.0"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-yy_coeff", "-1.0"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-zz_coeff", "-1.0"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-level", "1"); JX_CHECK_V_ERROR;
    jx_Parser_dhInsert(p, "-printStats", "0"); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH_2
}
