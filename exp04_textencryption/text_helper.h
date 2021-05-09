#ifndef TEXT_HELPER_H
#define TEXT_HELPER_H

#include <stdio.h>
#include<string.h>

char* ReadFile(const char *filename, int *read_size, int *string_size)
{
   char *buffer = NULL;
   // int string_size, read_size;
   FILE *handler = fopen(filename, "r");

   if (handler)
   {
       // Seek the last byte of the file
       fseek(handler, 0, SEEK_END);
       // Offset from the first to the last byte, or in other words, filesize
       *string_size = ftell(handler);
       // go back to the start of the file
       rewind(handler);

       // Allocate a string that can hold it all
       buffer = (char*) malloc(sizeof(char) * (*string_size + 1) );

       // Read it all in one operation
       *read_size = fread(buffer, sizeof(char), *string_size, handler);

       // fread doesn't set it so put a \0 in the last position
       // and buffer is now officially a string
       buffer[*read_size] = '\0';

       // Always remember to close the file.
       fclose(handler);
    }

    return buffer;
}

int WriteFile(const char *filename, const char *buffer, const int string_size)
{
   // int string_size, read_size;
   FILE *handler = fopen(filename, "w+");

   if (handler)
   {
       fwrite(buffer, sizeof(char), string_size, handler);
       fclose(handler);
       return 1;
    }
    else
    {
        return -1;
    }
}

#endif