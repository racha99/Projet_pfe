from django.contrib import admin


from .models import User,Teacher,Student,Comment,Module,comment_traité

from django.contrib.auth.admin import UserAdmin


class CustomUserAdmin(UserAdmin):
    fieldsets = (
        *UserAdmin.fieldsets,  # original form fieldsets, expanded
        (                      # new fieldset added on to the bottom
            'Custom Field Heading',  # group heading of your choice; set to None for a blank space instead of a header
            {
                'fields': (
                      "is_teacher",
                      "is_student", 
                      "is_admin",
                      "is_app_admin" 
                ),
            },
        ),
    )


admin.site.register(User, CustomUserAdmin)

admin.site.register(comment_traité)
admin.site.register(Teacher)
admin.site.register(Student)

admin.site.register(Comment)
admin.site.register(Module)
